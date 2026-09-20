import os
import unittest
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

from src.pipeline.models import RAGResult
from src.generation.models import GenerationResult
from src.pipeline.exceptions import PipelineError
from src.retrieval.models import RetrievedChunk

class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        self.patcher_env = patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"})
        self.patcher_env.start()

        # Clear Streamlit cache to ensure get_pipeline runs again
        import streamlit as st
        st.cache_resource.clear()

        patch("src.retrieval.config.load_config").start()
        patch("src.generation.config.load_config").start()
        patch("src.retrieval.retriever.Retriever").start()
        patch("src.generation.generator.GeminiGenerator").start()

        self.patcher_pipeline = patch("src.pipeline.rag_pipeline.RAGPipeline")
        self.mock_pipeline_cls = self.patcher_pipeline.start()

        self.mock_pipeline = MagicMock()
        self.mock_pipeline_cls.return_value = self.mock_pipeline

    def tearDown(self):
        patch.stopall()

    def test_app_loads(self):
        at = AppTest.from_file("../../src/app/streamlit_app.py")
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "A Comparative Study of Document Chunking Strategies in Retrieval-Augmented Generation for Educational Question Answering")

    def test_empty_question_rejected(self):
        at = AppTest.from_file("../../src/app/streamlit_app.py")
        at.run()
        at.sidebar.radio[0].set_value("Live RAG Demo").run()
        at.button[0].click().run()
        self.assertTrue(at.warning)
        self.assertEqual(at.warning[0].value, "Please enter a non-empty question.")
        self.mock_pipeline.run.assert_not_called()

    def test_successful_run(self):
        mock_chunk = RetrievedChunk(
            chunk_id="chunk1",
            doc_id="doc1",
            strategy="baseline_fixed_size",
            text="chunk text here",
            distance=0.1,
            char_start=0,
            char_end=15,
            provenance_spans=[{"page": 1}],
            metadata={}
        )
        mock_gen_result = MagicMock(spec=GenerationResult)
        mock_gen_result.answer = "This is the answer."
        mock_gen_result.retrieved_chunks = [mock_chunk]

        mock_rag_result = MagicMock(spec=RAGResult)
        mock_rag_result.generation = mock_gen_result

        self.mock_pipeline.run.return_value = mock_rag_result

        at = AppTest.from_file("../../src/app/streamlit_app.py")
        at.run()

        at.sidebar.radio[0].set_value("Live RAG Demo").run()

        at.text_area[0].input("What is chunking?").run()
        at.selectbox[0].select("eluc_section_aware").run()
        at.button[0].click().run()

        self.assertFalse(at.exception)

        self.mock_pipeline.run.assert_called_once_with(
            question="What is chunking?",
            strategy="eluc_section_aware"
        )

        infos = [m.value for m in at.info]
        self.assertIn("This is the answer.", infos)

        expanders = at.expander
        self.assertEqual(len(expanders), 1)
        self.assertIn("Rank 1 - ID: chunk1", expanders[0].label)

        chunk_markdowns = [m.value for m in expanders[0].markdown]
        self.assertIn("**Provenance:** `[{'page': 1}]`", chunk_markdowns)

        self.assertEqual(expanders[0].text[0].value, "chunk text here")

    def test_pipeline_error_handled(self):
        self.mock_pipeline.run.side_effect = PipelineError("retrieval failed")

        at = AppTest.from_file("../../src/app/streamlit_app.py")
        at.run()

        at.sidebar.radio[0].set_value("Live RAG Demo").run()
        at.text_area[0].input("test?").run()
        at.button[0].click().run()

        self.assertTrue(at.error)
        self.assertIn("Pipeline error: retrieval failed", [e.value for e in at.error])

    def test_unexpected_error_propagates(self):
        self.mock_pipeline.run.side_effect = TypeError("weird type")

        at = AppTest.from_file("../../src/app/streamlit_app.py")
        at.run()

        at.sidebar.radio[0].set_value("Live RAG Demo").run()
        at.text_area[0].input("test?").run()
        at.button[0].click().run()

        self.assertTrue(at.exception)
        self.assertIn("weird type", at.exception[0].value)

if __name__ == "__main__":
    unittest.main()

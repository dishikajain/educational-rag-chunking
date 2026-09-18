import unittest
from unittest.mock import MagicMock
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.models import RAGResult
from src.pipeline.exceptions import PipelineError
from src.generation.models import GenerationResult

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_generator = MagicMock()
        self.pipeline = RAGPipeline(retriever=self.mock_retriever, generator=self.mock_generator)

    def test_successful_run(self):
        question = "What is chunking?"
        strategy = "eluc_section_aware"
        mock_chunks = [MagicMock()]
        
        self.mock_retriever.retrieve.return_value = mock_chunks
        
        mock_gen_result = MagicMock(spec=GenerationResult)
        self.mock_generator.generate.return_value = mock_gen_result
        
        result = self.pipeline.run(question, strategy)
        
        self.mock_retriever.retrieve.assert_called_once_with(question, strategy)
        self.mock_generator.generate.assert_called_once_with(question, mock_chunks)
        
        # Explicitly verify the Generator receives the EXACT SAME list object
        generator_call = self.mock_generator.generate.call_args
        self.assertIs(generator_call.args[1], mock_chunks)
        
        # Explicitly verify inputs were not mutated
        self.assertEqual(question, "What is chunking?")
        self.assertEqual(strategy, "eluc_section_aware")
        self.assertEqual(len(mock_chunks), 1)
        
        self.assertIsInstance(result, RAGResult)
        self.assertEqual(result.question, question)
        self.assertEqual(result.strategy, strategy)
        self.assertEqual(result.generation, mock_gen_result)

    def test_empty_chunks_pass_through(self):
        question = "What is chunking?"
        strategy = "eluc_section_aware"
        mock_chunks = []
        
        self.mock_retriever.retrieve.return_value = mock_chunks
        
        mock_gen_result = MagicMock(spec=GenerationResult)
        self.mock_generator.generate.return_value = mock_gen_result
        
        result = self.pipeline.run(question, strategy)
        
        self.mock_generator.generate.assert_called_once_with(question, mock_chunks)
        
        # Explicitly verify the Generator receives the EXACT SAME list object
        generator_call = self.mock_generator.generate.call_args
        self.assertIs(generator_call.args[1], mock_chunks)
        
        self.assertEqual(result.generation, mock_gen_result)

    def test_invalid_question(self):
        for invalid_q in ["", "   ", None, 123]:
            with self.subTest(question=invalid_q):
                with self.assertRaises(ValueError):
                    self.pipeline.run(invalid_q, "strategy")
                    
    def test_invalid_strategy(self):
        for invalid_s in ["", "   ", None, 123]:
            with self.subTest(strategy=invalid_s):
                with self.assertRaises(ValueError):
                    self.pipeline.run("valid question", invalid_s)

    def test_retrieval_value_error_wrapped(self):
        self.mock_retriever.retrieve.side_effect = ValueError("retrieval failed")
        with self.assertRaises(PipelineError) as cm:
            self.pipeline.run("question", "strategy")
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_retrieval_runtime_error_wrapped(self):
        self.mock_retriever.retrieve.side_effect = RuntimeError("retrieval failed")
        with self.assertRaises(PipelineError) as cm:
            self.pipeline.run("question", "strategy")
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)

    def test_generation_value_error_wrapped(self):
        self.mock_retriever.retrieve.return_value = []
        self.mock_generator.generate.side_effect = ValueError("generation failed")
        with self.assertRaises(PipelineError) as cm:
            self.pipeline.run("question", "strategy")
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_generation_runtime_error_wrapped(self):
        self.mock_retriever.retrieve.return_value = []
        self.mock_generator.generate.side_effect = RuntimeError("generation failed")
        with self.assertRaises(PipelineError) as cm:
            self.pipeline.run("question", "strategy")
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        
    def test_unexpected_type_error_propagates(self):
        self.mock_retriever.retrieve.side_effect = TypeError("unexpected type")
        with self.assertRaises(TypeError):
            self.pipeline.run("question", "strategy")
            
    def test_unexpected_key_error_propagates(self):
        self.mock_retriever.retrieve.side_effect = KeyError("unexpected key")
        with self.assertRaises(KeyError):
            self.pipeline.run("question", "strategy")

    def test_generator_unexpected_type_error_propagates(self):
        self.mock_retriever.retrieve.return_value = []
        self.mock_generator.generate.side_effect = TypeError("unexpected type")
        with self.assertRaises(TypeError):
            self.pipeline.run("question", "strategy")
            
    def test_generator_unexpected_key_error_propagates(self):
        self.mock_retriever.retrieve.return_value = []
        self.mock_generator.generate.side_effect = KeyError("unexpected key")
        with self.assertRaises(KeyError):
            self.pipeline.run("question", "strategy")

if __name__ == '__main__':
    unittest.main()

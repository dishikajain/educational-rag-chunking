import pytest
import os
import tempfile
import json
import math
import chromadb
from unittest.mock import patch, MagicMock
from src.retrieval.config import RetrievalConfig, load_config
from src.retrieval.models import RetrievedChunk
from src.retrieval.retriever import Retriever
import numpy as np

@pytest.fixture
def mock_config():
    return RetrievalConfig(
        model_name="BAAI/bge-base-en-v1.5",
        model_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        query_instruction="Represent this sentence for searching relevant passages: ",
        top_k=5,
        distance_metric="cosine",
        batch_size=1,
        storage_path="data/chroma_db"
    )

@pytest.fixture
def temp_db():
    client = chromadb.EphemeralClient()
    baseline_col = client.get_or_create_collection("baseline_fixed_size_bge_base_en_v1_5")
    client.delete_collection("baseline_fixed_size_bge_base_en_v1_5")
        
    eluc_col = client.get_or_create_collection("eluc_section_aware_bge_base_en_v1_5")
    client.delete_collection("eluc_section_aware_bge_base_en_v1_5")

    baseline_col = client.create_collection("baseline_fixed_size_bge_base_en_v1_5")
    baseline_col.add(
        ids=["c1", "c2", "c3"],
        embeddings=[[0.1]*768, [0.2]*768, [0.3]*768],
        documents=["text1", "text2", "text3"],
        metadatas=[
            {"doc_id": "d1", "char_start": 0, "char_end": 5, "provenance_spans": json.dumps([{"char_start":0, "char_end":5, "label":"P"}])},
            {"doc_id": "d1", "char_start": 5, "char_end": 10, "provenance_spans": json.dumps([{"char_start":5, "char_end":10, "label":"P"}])},
            {"doc_id": "d2", "char_start": 0, "char_end": 5, "provenance_spans": json.dumps([{"char_start":0, "char_end":5, "label":"P"}])}
        ]
    )
    
    eluc_col = client.create_collection("eluc_section_aware_bge_base_en_v1_5")
    eluc_col.add(
        ids=["e1", "e2"],
        embeddings=[[0.1]*768, [0.2]*768],
        documents=["eluc1", "eluc2"],
        metadatas=[
            {"doc_id": "d1", "char_start": 0, "char_end": 5, "provenance_spans": json.dumps([{"char_start":0, "char_end":5, "label":"P"}])},
            {"doc_id": "d1", "char_start": 5, "char_end": 10, "provenance_spans": json.dumps([{"char_start":5, "char_end":10, "label":"P"}])}
        ]
    )
    yield client
    client.clear_system_cache()


def test_model_loads_with_exact_revision(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        retriever = Retriever(mock_config, client=temp_db)
        mock_st.assert_called_once_with(
            model_name_or_path="BAAI/bge-base-en-v1.5",
            revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
        )


def test_query_instruction_added_exactly_once(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*768)
        
        retriever = Retriever(mock_config, client=temp_db)
        query = "What is RAG?"
        retriever.retrieve(query, "baseline_fixed_size")
        
        expected_call_query = mock_config.query_instruction + query
        mock_instance.encode.assert_called_once_with(
            expected_call_query,
            normalize_embeddings=True,
            show_progress_bar=False
        )

def test_empty_or_whitespace_query_fails(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer'):
        retriever = Retriever(mock_config, client=temp_db)
        with pytest.raises(ValueError, match="empty or whitespace"):
            retriever.retrieve("", "baseline_fixed_size")
        with pytest.raises(ValueError, match="empty or whitespace"):
            retriever.retrieve("   \t\n", "baseline_fixed_size")
        with pytest.raises(TypeError, match="must be a string"):
            retriever.retrieve(None, "baseline_fixed_size") # type: ignore


def test_missing_collection_fails_clearly(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*768)
        
        retriever = Retriever(mock_config, client=temp_db)
        temp_db.delete_collection("baseline_fixed_size_bge_base_en_v1_5")
        with pytest.raises(ValueError, match="not found for strategy baseline_fixed_size"):
            retriever.retrieve("test", "baseline_fixed_size")

def test_unknown_strategy_fails(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer'):
        retriever = Retriever(mock_config, client=temp_db)
        with pytest.raises(ValueError, match="Unknown strategy: unknown_strategy"):
            retriever.retrieve("test", "unknown_strategy")


def test_same_query_identical_embedding_for_strategies(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*768)
        
        retriever = Retriever(mock_config, client=temp_db)
        
        # We need to spy on the collection query to ensure identical embeddings are passed to ChromaDB
        query_embeddings_passed = []
        original_get_collection = temp_db.get_collection
        
        def spy_get_collection(name):
            col = original_get_collection(name=name)
            original_query = col.query
            
            def spy_query(*args, **kwargs):
                query_embeddings_passed.append(kwargs.get('query_embeddings') or args[0])
                return original_query(*args, **kwargs)
                
            mock_col = MagicMock(wraps=col)
            mock_col.count.side_effect = col.count
            mock_col.query.side_effect = spy_query
            return mock_col

        with patch.object(temp_db, 'get_collection', side_effect=spy_get_collection):
            retriever.retrieve("test query", "baseline_fixed_size")
            retriever.retrieve("test query", "eluc_section_aware")
            
            assert len(query_embeddings_passed) == 2
            assert query_embeddings_passed[0] == query_embeddings_passed[1]


def test_output_integrity(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*768)
        
        retriever = Retriever(mock_config, client=temp_db)
        results = retriever.retrieve("test", "baseline_fixed_size")
        
        assert len(results) > 0
        chunk = results[0]
        
        # Test 11, 12, 13
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.char_start, int)
        assert isinstance(chunk.char_end, int)
        assert isinstance(chunk.provenance_spans, list)
        
        if chunk.chunk_id == "c1":
            assert chunk.text == "text1"
            assert chunk.char_start == 0
            assert chunk.char_end == 5
            assert chunk.provenance_spans == [{"char_start": 0, "char_end": 5, "label": "P"}]


def test_distance_is_numeric_and_finite(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*768)
        
        retriever = Retriever(mock_config, client=temp_db)
        results = retriever.retrieve("test", "baseline_fixed_size")
        
        for chunk in results:
            assert isinstance(chunk.distance, float)
            assert math.isfinite(chunk.distance)

def test_top_k_respected(mock_config, temp_db):
    mock_config.top_k = 2
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*768)
        
        retriever = Retriever(mock_config, client=temp_db)
        results = retriever.retrieve("test", "baseline_fixed_size")
        assert len(results) == 2


def test_n_results_does_not_exceed_collection_size(mock_config, temp_db):
    mock_config.top_k = 100
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*768)
        
        retriever = Retriever(mock_config, client=temp_db)
        results = retriever.retrieve("test", "baseline_fixed_size")
        assert len(results) == 3 # the collection only has 3 items


def test_retrieval_does_not_modify_data(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*768)
        
        retriever = Retriever(mock_config, client=temp_db)
        
        col_before = temp_db.get_collection("baseline_fixed_size_bge_base_en_v1_5")
        count_before = col_before.count()
        
        retriever.retrieve("test", "baseline_fixed_size")
        
        col_after = temp_db.get_collection("baseline_fixed_size_bge_base_en_v1_5")
        assert col_after.count() == count_before
        
        # Verify text was not mutated by checking if they are identical
        items = col_after.get(ids=["c1"])
        assert items["documents"][0] == "text1"


def test_query_dimension_check(mock_config, temp_db):
    with patch('src.retrieval.retriever.SentenceTransformer') as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([0.1]*384) # Invalid dimension
        
        retriever = Retriever(mock_config, client=temp_db)
        with pytest.raises(ValueError, match="expected 768"):
            retriever.retrieve("test", "baseline_fixed_size")


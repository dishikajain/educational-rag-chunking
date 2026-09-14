import pytest
import os
import tempfile
import json
import chromadb
from src.embedding.models import EmbeddedChunkDocument, EmbeddedChunk
from src.preprocessing.models import ProvenanceSpan
from src.indexing.config import IndexingConfig
from src.indexing.indexer import index_document, sanitize_collection_name

@pytest.fixture
def temp_db():
    client = chromadb.EphemeralClient()
    yield client
    client.clear_system_cache()

@pytest.fixture
def sample_doc():
    spans = [ProvenanceSpan(char_start=0, char_end=10, source_blocks=[(1, 0)], label="P")]
    return EmbeddedChunkDocument(
        doc_id="test_doc",
        strategy="baseline_fixed_size",
        model_name="BAAI/bge-base-en-v1.5",
        model_revision="hash123",
        embedding_dimension=768,
        chunks=[
            EmbeddedChunk(
                chunk_id="c1",
                doc_id="test_doc",
                text="Hello world",
                strategy="baseline_fixed_size",
                char_start=0,
                char_end=10,
                provenance_spans=spans,
                embedding=[0.1]*768
            )
        ]
    )

def test_successful_indexing(sample_doc, temp_db):
    config = IndexingConfig(embedding_dimension=768)
    report = index_document(sample_doc, config, client=temp_db)
    
    assert report.chunks_inserted == 1
    
    collection = temp_db.get_collection(report.collection_name)
    assert collection.count() == 1
    
    result = collection.get(ids=["c1"], include=["embeddings", "documents", "metadatas"])
    assert result["documents"][0] == "Hello world"
    assert len(result["embeddings"][0]) == 768
    
    meta = result["metadatas"][0]
    assert meta["doc_id"] == "test_doc"
    assert meta["char_start"] == 0
    assert meta["chunk_index"] == 0
    assert meta["model_name"] == "BAAI/bge-base-en-v1.5"
    assert meta["model_revision"] == "hash123"
    
    # Check Collection Metadata
    col_meta = collection.metadata
    assert col_meta["hnsw:space"] == "cosine"
    assert col_meta["embedding_dimension"] == 768
    assert col_meta["strategy"] == "baseline_fixed_size"
    
    # Provenance JSON round-trip
    prov_str = meta["provenance_spans"]
    prov_loaded = json.loads(prov_str)
    assert len(prov_loaded) == 1
    assert prov_loaded[0] == {"char_start": 0, "char_end": 10, "source_blocks": [[1, 0]], "label": "P"}


def test_dimension_validation_fast_fail(sample_doc, temp_db):
    with pytest.raises(ValueError, match="embedding_dimension must be exactly 768"):
        config = IndexingConfig(embedding_dimension=384) 
        
    config = IndexingConfig(embedding_dimension=768)
    sample_doc.embedding_dimension = 384 
    with pytest.raises(ValueError, match="Document dimension must be 768"):
        index_document(sample_doc, config, client=temp_db)
        
    sample_doc.embedding_dimension = 768
    sample_doc.chunks[0].embedding = [0.1]*10 
    with pytest.raises(ValueError, match="invalid dimension"):
        index_document(sample_doc, config, client=temp_db)

    sample_doc.chunks[0].embedding = None
    with pytest.raises(ValueError, match="None embedding"):
        index_document(sample_doc, config, client=temp_db)

    sample_doc.chunks[0].embedding = [0.1]*767 + [float('nan')]
    with pytest.raises(ValueError, match="non-finite value"):
        index_document(sample_doc, config, client=temp_db)

    sample_doc.chunks[0].embedding = [0.1]*767 + [float('inf')]
    with pytest.raises(ValueError, match="non-finite value"):
        index_document(sample_doc, config, client=temp_db)

    sample_doc.chunks[0].embedding = [0.1]*767 + ["string"]
    with pytest.raises(TypeError, match="non-numeric value"):
        index_document(sample_doc, config, client=temp_db)

def test_upsert_idempotency(sample_doc, temp_db):
    config = IndexingConfig(reset_collection_on_run=False)
    index_document(sample_doc, config, client=temp_db)
    report = index_document(sample_doc, config, client=temp_db)
    
    collection = temp_db.get_collection(report.collection_name)
    assert collection.count() == 1 

def test_empty_document_behavior(sample_doc, temp_db):
    sample_doc.chunks = []
    config = IndexingConfig()
    with pytest.raises(ValueError, match="Cannot index an empty document"):
        index_document(sample_doc, config, client=temp_db)

def test_separate_collections(sample_doc, temp_db):
    config = IndexingConfig()
    report1 = index_document(sample_doc, config, client=temp_db)
    
    import copy
    eluc_doc = copy.deepcopy(sample_doc)
    eluc_doc.strategy = "eluc_section_aware"
    eluc_doc.chunks[0].chunk_id = "c2"
    
    report2 = index_document(eluc_doc, config, client=temp_db)
    
    assert report1.collection_name != report2.collection_name
    
def test_targeted_reset(sample_doc, temp_db):
    config = IndexingConfig(reset_collection_on_run=True)
    report1 = index_document(sample_doc, config, client=temp_db)
    
    col2 = temp_db.create_collection("unrelated_col")
    col2.add(ids=["1"], embeddings=[[0.0]*768], documents=["test"])
    
    sample_doc.chunks[0].text = "New Data"
    index_document(sample_doc, config, client=temp_db)
    
    assert temp_db.get_collection("unrelated_col").count() == 1
    
    col1 = temp_db.get_collection(report1.collection_name)
    assert col1.count() == 1
    result = col1.get(ids=["c1"])
    assert result["documents"][0] == "New Data"

def test_collection_naming_determinism():
    name = sanitize_collection_name("baseline", "BAAI/bge-base-en-v1.5")
    assert name == "baseline_bge_base_en_v1_5"

def test_invalid_config_values():
    with pytest.raises(ValueError, match="must be exactly 768"):
        IndexingConfig(embedding_dimension=100)
    with pytest.raises(ValueError, match="must be 'cosine'"):
        IndexingConfig(distance_metric="l2")
    with pytest.raises(ValueError, match="must be exactly 256"):
        IndexingConfig(index_batch_size=128)
    with pytest.raises(ValueError, match="must be boolean"):
        IndexingConfig(reset_collection_on_run="true")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        IndexingConfig(storage_path="   ")

def test_missing_yaml_config():
    with pytest.raises(FileNotFoundError, match="Indexing configuration file not found"):
        IndexingConfig.from_yaml("nonexistent_file.yaml")

import json
import logging
import re
from typing import Optional
import chromadb
from dataclasses import asdict
from src.embedding.models import EmbeddedChunkDocument
from src.indexing.config import IndexingConfig
from src.indexing.models import IndexingReport

logger = logging.getLogger(__name__)

def sanitize_collection_name(strategy: str, model_name: str) -> str:
    # Format: {strategy}_{model_basename}
    model_basename = model_name.split("/")[-1]
    raw_name = f"{strategy}_{model_basename}"
    safe_name = "".join(c if c.isalnum() else "_" for c in raw_name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")
    if len(safe_name) < 3:
        safe_name = safe_name.ljust(3, "_")
    return safe_name[:63]

def index_document(
    doc: EmbeddedChunkDocument, 
    config: IndexingConfig, 
    client: Optional[chromadb.ClientAPI] = None
) -> IndexingReport:
    
    if config.embedding_dimension != 768:
        raise ValueError(f"Configured dimension must be 768, got {config.embedding_dimension}")
        
    if doc.embedding_dimension != 768:
        raise ValueError(f"Document dimension must be 768, got {doc.embedding_dimension}")

    if not doc.chunks:
        raise ValueError("Cannot index an empty document in experimental pipeline (FAIL FAST).")
        
    import math
    for chunk in doc.chunks:
        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk.chunk_id} has None embedding.")
        if not hasattr(chunk.embedding, "__len__") or len(chunk.embedding) != 768:
            raise ValueError(f"Chunk {chunk.chunk_id} has invalid dimension.")
        for val in chunk.embedding:
            if not isinstance(val, (int, float)):
                raise TypeError(f"Chunk {chunk.chunk_id} embedding contains non-numeric value: {type(val)}")
            if not math.isfinite(val):
                raise ValueError(f"Chunk {chunk.chunk_id} embedding contains non-finite value: {val}")

    if client is None:
        client = chromadb.PersistentClient(path=config.storage_path)

    collection_name = sanitize_collection_name(doc.strategy, doc.model_name)
    
    if config.reset_collection_on_run:
        try:
            client.get_collection(name=collection_name)
            client.delete_collection(name=collection_name)
        except chromadb.errors.InvalidCollectionException:
            pass

    collection_metadata = {
        "strategy": doc.strategy,
        "model_name": doc.model_name,
        "model_revision": doc.model_revision,
        "embedding_dimension": 768,
        "distance_metric": config.distance_metric
    }

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": config.distance_metric, **collection_metadata}
    )

    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(doc.chunks):
        ids.append(chunk.chunk_id)
        embeddings.append(chunk.embedding)
        documents.append(chunk.text)
        
        try:
            prov_list = [asdict(span) for span in chunk.provenance_spans]
        except TypeError:
            prov_list = [span.__dict__ if hasattr(span, '__dict__') else span for span in chunk.provenance_spans]
            
        prov_json = json.dumps(prov_list)
        
        metadatas.append({
            "doc_id": chunk.doc_id,
            "strategy": chunk.strategy,
            "chunk_index": i,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "model_name": doc.model_name,
            "model_revision": doc.model_revision,
            "provenance_spans": prov_json
        })
        
    batch_size = config.index_batch_size
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    return IndexingReport(
        collection_name=collection_name,
        chunks_inserted=len(ids),
        dimension=768
    )

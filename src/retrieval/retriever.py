import json
import logging
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.config import RetrievalConfig
from src.retrieval.models import RetrievedChunk
from src.indexing.indexer import sanitize_collection_name

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, config: RetrievalConfig, client: Optional[chromadb.ClientAPI] = None):
        self.config = config
        
        # Load the frozen model
        self.model = SentenceTransformer(
            model_name_or_path=self.config.model_name,
            revision=self.config.model_revision
        )
        
        # We need to ensure we return 768 dim, BGE models do this natively, 
        # but just standard practice for our pipeline.
        
        if client is None:
            self.client = chromadb.PersistentClient(path=self.config.storage_path)
        else:
            self.client = client
            
    def _validate_query(self, query: str):
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
            
    def retrieve(self, query: str, strategy: str) -> List[RetrievedChunk]:
        self._validate_query(query)
        
        if strategy not in ["baseline_fixed_size", "eluc_section_aware"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply the frozen BGE instruction only for embedding
        query_to_embed = self.config.query_instruction + query
        
        # Embed the query
        query_embedding = self.model.encode(
            query_to_embed,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Ensure dimension is exactly 768 and values are numeric/finite
        import math
        if len(query_embedding) != 768:
            raise ValueError(f"Query embedding dimension is {len(query_embedding)}, expected 768")
        for val in query_embedding:
            try:
                f_val = float(val)
            except (TypeError, ValueError):
                raise TypeError("Query embedding contains non-numeric value")
            if not math.isfinite(f_val):
                raise ValueError("Query embedding contains non-finite value")
            
        collection_name = sanitize_collection_name(strategy, self.config.model_name)
        
        try:
            collection = self.client.get_collection(name=collection_name)
        except chromadb.errors.InvalidCollectionException as e:
            raise ValueError(f"Collection {collection_name} not found for strategy {strategy}") from e
            
        # Ensure we don't request more results than exist in the collection
        n_results = min(self.config.top_k, collection.count())
        if n_results == 0:
            return []
            
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_chunks = []
        
        # ChromaDB returns lists of lists (one per query)
        if not results["ids"] or not results["ids"][0]:
            return []
            
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        # Construct RetrievedChunk objects
        for i in range(len(ids)):
            meta = metadatas[i]
            
            # Decode provenance_spans from JSON
            try:
                provenance_spans = json.loads(meta["provenance_spans"])
            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError("Missing or invalid provenance_spans metadata") from e
                
            chunk = RetrievedChunk(
                chunk_id=ids[i],
                doc_id=meta.get("doc_id", ""),
                strategy=strategy,
                text=documents[i],
                distance=float(distances[i]),
                char_start=meta.get("char_start", 0),
                char_end=meta.get("char_end", 0),
                provenance_spans=provenance_spans,
                metadata={k: v for k, v in meta.items() if k not in ["provenance_spans", "doc_id", "char_start", "char_end"]}
            )
            retrieved_chunks.append(chunk)
            
        return retrieved_chunks

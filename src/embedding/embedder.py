import logging
from typing import Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is not installed. Please install it."
    ) from e

from src.chunking.models import BaselineChunkDocument
from src.eluc.models import ELUCChunkDocument
from src.embedding.config import EmbeddingConfig
from src.embedding.models import EmbeddedChunk, EmbeddedChunkDocument


logger = logging.getLogger(__name__)


def embed_document(
    doc: Union[BaselineChunkDocument, ELUCChunkDocument],
    config: EmbeddingConfig,
) -> EmbeddedChunkDocument:

    try:
        model = SentenceTransformer(
            model_name_or_path=config.model_name,
            revision=config.model_revision,
        )
    except Exception as e:
        logger.error(
            f"Failed to load model {config.model_name} "
            f"at revision {config.model_revision}: {e}"
        )
        raise RuntimeError(f"Model loading failed: {e}") from e

    actual_dim = model.get_sentence_embedding_dimension()

    if actual_dim != config.embedding_dimension:
        raise ValueError(
            f"Model {config.model_name} produced dimension {actual_dim}, "
            f"but config requires {config.embedding_dimension}."
        )

    max_seq_length = model.max_seq_length

    texts_to_embed = []
    chunk_mapping = []

    for chunk in doc.chunks:
        text = chunk.text

        if not text or not text.strip():
            raise ValueError(
                f"Empty or whitespace-only chunk encountered: "
                f"chunk_id={chunk.chunk_id}"
            )

        tokens = model.tokenizer(
            text,
            add_special_tokens=False,
        )

        token_len = len(tokens["input_ids"])

        if (
            max_seq_length is not None
            and token_len > max_seq_length
        ):
            logger.warning(
                f"Chunk {chunk.chunk_id} token length ({token_len}) "
                f"exceeds model max_seq_length ({max_seq_length}). "
                f"The input will be truncated."
            )

        texts_to_embed.append(text)
        chunk_mapping.append(chunk)

    if not texts_to_embed:
        return EmbeddedChunkDocument(
            doc_id=doc.doc_id,
            strategy=doc.strategy,
            model_name=config.model_name,
            model_revision=config.model_revision,
            embedding_dimension=config.embedding_dimension,
            chunks=[],
        )

    try:
        processing_kwargs = {
            "text": {
                "truncation": True,
            }
        }

        if max_seq_length is not None:
            processing_kwargs["text"]["max_length"] = max_seq_length

        embeddings = model.encode(
            texts_to_embed,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
            processing_kwargs=processing_kwargs,
        )

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise RuntimeError(
            f"Embedding generation failed: {e}"
        ) from e

    embeddings = np.asarray(embeddings)

    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError(
            "Model produced NaN or Inf embeddings. Failing fast."
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    embeddings_list = embeddings.tolist()

    embedded_chunks = []

    for chunk, embedding in zip(
        chunk_mapping,
        embeddings_list,
    ):
        embedded_chunk = EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            embedding=embedding,
            strategy=doc.strategy,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            provenance_spans=chunk.provenance_spans,
        )

        embedded_chunks.append(embedded_chunk)

    return EmbeddedChunkDocument(
        doc_id=doc.doc_id,
        strategy=doc.strategy,
        model_name=config.model_name,
        model_revision=config.model_revision,
        embedding_dimension=config.embedding_dimension,
        chunks=embedded_chunks,
    )
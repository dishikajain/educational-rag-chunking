import logging
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    import src.embedding.embedder
    monkeypatch.setattr(src.embedding.embedder, "SentenceTransformer", MockSentenceTransformer)


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    import src.embedding.embedder
    monkeypatch.setattr(src.embedding.embedder, "SentenceTransformer", MockSentenceTransformer)


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    import src.embedding.embedder
    monkeypatch.setattr(src.embedding.embedder, "SentenceTransformer", MockSentenceTransformer)


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    import src.embedding.embedder
    monkeypatch.setattr(src.embedding.embedder, "SentenceTransformer", MockSentenceTransformer)


class MockSentenceTransformer:
    def __init__(self, model_name_or_path, revision=None, **kwargs):
        self.max_seq_length = 512
        self.model_name_or_path = model_name_or_path
        self.revision = revision

        self.tokenizer = lambda x, add_special_tokens=False, **kw: {
            "input_ids": [1] * (600 if "long" in x else 3)
        }

        MockSentenceTransformer.last_instance = self

        if "invalid" in model_name_or_path:
            raise RuntimeError("Invalid model")

    def get_sentence_embedding_dimension(self):
        return 768

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=False,
        convert_to_numpy=True,
        **kwargs
    ):
        self.last_encode_kwargs = {
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "convert_to_numpy": convert_to_numpy,
            "processing_kwargs": kwargs.get("processing_kwargs"),
        }

        self.last_texts = texts

        embeddings = []

        for text in texts:
            val = float(sum(ord(c) for c in text)) / 1000.0

            vec = np.full(
                768,
                val,
                dtype=np.float32
            )

            if normalize_embeddings:
                norm = np.linalg.norm(vec)

                if norm > 0:
                    vec = vec / norm

            embeddings.append(vec)

        return np.array(
            embeddings,
            dtype=np.float32
        )





from src.preprocessing.models import ProvenanceSpan
from src.chunking.models import (
    BaselineChunkDocument,
    BaselineChunk,
)
from src.eluc.models import (
    ELUCChunkDocument,
    ELUCChunk,
)
from src.embedding.config import EmbeddingConfig
from src.embedding.embedder import embed_document


@pytest.fixture
def config():
    return EmbeddingConfig(
        model_name="BAAI/bge-base-en-v1.5",
        model_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        embedding_dimension=768,
        normalize_embeddings=True,
        batch_size=32
    )


@pytest.fixture
def dummy_provenance():
    return [
        ProvenanceSpan(
            char_start=0,
            char_end=10,
            source_blocks=[(1, 0)],
            label="BODY"
        )
    ]


@pytest.fixture
def baseline_doc(dummy_provenance):
    chunks = [
        BaselineChunk(
            chunk_id="c1",
            doc_id="doc1",
            chunk_index=0,
            text="This is a test chunk.",
            char_start=0,
            char_end=21,
            provenance_spans=dummy_provenance
        ),
        BaselineChunk(
            chunk_id="c2",
            doc_id="doc1",
            chunk_index=1,
            text="Another test chunk for embeddings.",
            char_start=22,
            char_end=56,
            provenance_spans=dummy_provenance
        )
    ]

    return BaselineChunkDocument(
        doc_id="doc1",
        strategy="baseline_fixed_size",
        config_used=None,
        chunks=chunks
    )


@pytest.fixture
def eluc_doc(dummy_provenance):
    chunks = [
        ELUCChunk(
            chunk_id="e1",
            doc_id="doc1",
            chunk_index=0,
            text="This is a test chunk.",
            char_start=0,
            char_end=21,
            provenance_spans=dummy_provenance,
            source_blocks=[(1, 0)],
            heading_label=None,
            contains_sidebar=False,
            block_count=1,
            page_span=(1, 1)
        )
    ]

    return ELUCChunkDocument(
        doc_id="doc1",
        strategy="eluc_section_aware",
        config_used=None,
        chunks=chunks
    )


def test_embedding_dimensionality(baseline_doc, config):
    result = embed_document(
        baseline_doc,
        config
    )

    assert result.embedding_dimension == 768

    for chunk in result.chunks:
        assert len(chunk.embedding) == 768


def test_deterministic_embedding(baseline_doc, config):
    res1 = embed_document(
        baseline_doc,
        config
    )

    res2 = embed_document(
        baseline_doc,
        config
    )

    emb1 = np.array(
        res1.chunks[0].embedding
    )

    emb2 = np.array(
        res2.chunks[0].embedding
    )

    np.testing.assert_allclose(
        emb1,
        emb2,
        atol=1e-5
    )


def test_metadata_preservation(baseline_doc, config):
    result = embed_document(
        baseline_doc,
        config
    )

    assert len(result.chunks) == 2

    orig_c1 = baseline_doc.chunks[0]
    res_c1 = result.chunks[0]

    assert res_c1.chunk_id == orig_c1.chunk_id
    assert res_c1.doc_id == orig_c1.doc_id
    assert res_c1.text == orig_c1.text
    assert res_c1.strategy == "baseline_fixed_size"
    assert res_c1.char_start == orig_c1.char_start
    assert res_c1.char_end == orig_c1.char_end
    assert res_c1.provenance_spans == orig_c1.provenance_spans


def test_fairness_same_text_equivalent_embeddings(
    baseline_doc,
    eluc_doc,
    config
):
    res_base = embed_document(
        baseline_doc,
        config
    )

    res_eluc = embed_document(
        eluc_doc,
        config
    )

    emb_base = np.array(
        res_base.chunks[0].embedding
    )

    emb_eluc = np.array(
        res_eluc.chunks[0].embedding
    )

    np.testing.assert_allclose(
        emb_base,
        emb_eluc,
        atol=1e-5
    )


def test_normalization(baseline_doc, config):
    config.normalize_embeddings = True

    result = embed_document(
        baseline_doc,
        config
    )

    for chunk in result.chunks:
        emb = np.array(
            chunk.embedding
        )

        norm = np.linalg.norm(emb)

        assert np.isclose(
            norm,
            1.0,
            atol=1e-5
        )


def test_empty_text_fails_fast(baseline_doc, config):
    baseline_doc.chunks[0].text = "   \n"

    with pytest.raises(
        ValueError,
        match="Empty or whitespace-only chunk encountered"
    ):
        embed_document(
            baseline_doc,
            config
        )


def test_invalid_model_fails_fast(
    baseline_doc,
    config
):
    config.model_name = "invalid-model-name/does-not-exist"

    with pytest.raises(
        RuntimeError,
        match="Model loading failed"
    ):
        embed_document(
            baseline_doc,
            config
        )


def test_wrong_dimensionality_fails_fast(
    baseline_doc,
    config
):
    config.embedding_dimension = 1024

    with pytest.raises(
        ValueError,
        match="produced dimension 768, but config requires 1024"
    ):
        embed_document(
            baseline_doc,
            config
        )


def test_nan_detection(
    monkeypatch,
    baseline_doc,
    config
):
    class NaNSentenceTransformer(
        MockSentenceTransformer
    ):
        def encode(self, *args, **kwargs):
            return np.full(
                (len(args[0]), 768),
                np.nan,
                dtype=np.float32
            )

    import src.embedding.embedder

    monkeypatch.setattr(
        src.embedding.embedder,
        "SentenceTransformer",
        NaNSentenceTransformer
    )

    with pytest.raises(
        ValueError,
        match="Model produced NaN or Inf embeddings"
    ):
        embed_document(
            baseline_doc,
            config
        )


def test_model_kwargs_and_text_preservation(
    baseline_doc,
    config
):
    embed_document(
        baseline_doc,
        config
    )

    instance = MockSentenceTransformer.last_instance

    assert (
        instance.model_name_or_path
        == "BAAI/bge-base-en-v1.5"
    )

    assert (
        instance.revision
        == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    )

    assert (
        instance.last_encode_kwargs["batch_size"]
        == 32
    )

    assert (
        instance.last_encode_kwargs["normalize_embeddings"]
        is True
    )

    assert (
        instance.last_encode_kwargs["convert_to_numpy"]
        is True
    )

    processing_kwargs = (
        instance.last_encode_kwargs["processing_kwargs"]
    )

    assert processing_kwargs is not None

    assert (
        processing_kwargs["text"]["truncation"]
        is True
    )

    assert (
        processing_kwargs["text"]["max_length"]
        == 512
    )

    assert (
        instance.last_texts[0]
        == "This is a test chunk."
    )

    assert (
        instance.last_texts[1]
        == "Another test chunk for embeddings."
    )


def test_truncation_logging(
    baseline_doc,
    config,
    caplog
):
    baseline_doc.chunks[0].text = (
        "This is a very long text that triggers "
        "the mock tokenizer."
    )

    with caplog.at_level(logging.WARNING):
        embed_document(
            baseline_doc,
            config
        )

    assert (
        "exceeds model max_seq_length"
        in caplog.text
    )
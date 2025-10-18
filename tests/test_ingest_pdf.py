from pathlib import Path
from types import SimpleNamespace

import pytest

import src.ingest as ingest_module


def make_fake_chroma(existing_ids=None):
    existing_ids = set(existing_ids or [])

    class FakeChroma:
        last_instance = None

        def __init__(
            self, embedding_function=None, persist_directory=None, collection_name=None
        ):
            self.embedding_function = embedding_function
            self.persist_directory = persist_directory
            self.collection_name = collection_name
            self.added_docs = []
            self.added_ids = []
            self.persisted = False
            self._existing_ids = set(existing_ids)
            FakeChroma.last_instance = self

        def get(self):
            return {"ids": list(self._existing_ids)}

        def add_documents(self, docs, ids=None):
            self.added_docs.extend(docs)
            self.added_ids.extend(ids or [])

        def persist(self):
            self.persisted = True

        def as_retriever(self):
            return "fake_retriever"

    return FakeChroma


@pytest.fixture
def synthetic_pdf_chunks():
    # Create synthetic page chunks that mimic what PDF extraction would return.
    return [
        {
            "page_content": "Page one content",
            "metadata": {"page": 1, "source": "synthetic"},
        },
        {
            "page_content": "",
            "metadata": {"page": 2, "source": "synthetic"},
        },  # empty page should be filtered
        {
            "page_content": "Page three content",
            "metadata": {"page": 3, "source": "synthetic"},
        },
    ]


def test_ingest_with_filtered_chunks(monkeypatch, tmp_path, synthetic_pdf_chunks):
    # Arrange: monkeypatch embedding and splitting logic
    dummy_embeddings = object()
    monkeypatch.setattr(
        ingest_module, "get_sentence_embeddings", lambda: dummy_embeddings
    )
    # ensure the ingest uses our synthetic chunks (skip actual loading/parsing)
    monkeypatch.setattr(
        ingest_module, "split_documents", lambda docs: synthetic_pdf_chunks
    )

    called = {}

    def fake_get_chroma_ingest(*, embeddings, chunks, directory, collection_name):
        called["embeddings"] = embeddings
        called["chunks"] = chunks
        called["directory"] = directory
        called["collection_name"] = collection_name
        return "fake_vectordb"

    monkeypatch.setattr(ingest_module, "get_chroma_ingest", fake_get_chroma_ingest)

    # Act
    persist_path = tmp_path / "storage" / "chroma_test"
    ingest_module.ingest(
        data_dir=tmp_path, persist_path=persist_path, collection_name="pdf_test"
    )

    # Assert: persist dir created
    assert persist_path.exists() and persist_path.is_dir()
    # get_chroma_ingest called with expected args
    assert called["embeddings"] is dummy_embeddings
    assert called["directory"] == str(persist_path)
    assert called["collection_name"] == "pdf_test"
    # only non-empty pages should be passed through
    passed_texts = [
        c["page_content"] for c in called["chunks"] if c["page_content"].strip()
    ]
    assert passed_texts == ["Page one content", "Page three content"]

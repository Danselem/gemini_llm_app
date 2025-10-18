from pathlib import Path
from types import SimpleNamespace

import pytest

import src.vectors.chroma_vector as chroma_vector


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


def _pdf_to_page_chunks(pdf_path: Path):
    import importlib

    pdf_module = None
    for name in ("pypdf", "PyPDF2"):
        try:
            pdf_module = importlib.import_module(name)
            break
        except ImportError:
            continue

    if pdf_module is None:
        pytest.skip("requires 'pypdf' or 'PyPDF2' to extract PDF text")

    reader = pdf_module.PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        # prefer extract_text(), fallback to get_text() for compatibility
        if hasattr(page, "extract_text"):
            text = page.extract_text() or ""
        elif hasattr(page, "get_text"):
            text = page.get_text() or ""
        else:
            text = ""
        pages.append(
            {
                "page_content": text,
                "metadata": {"page": i + 1, "source": str(pdf_path)},
            }
        )
    return pages


def test_get_chroma_ingest_from_pdf(monkeypatch, tmp_path):
    """
    Uses data/pdfs/p554.pdf to build page chunks and ensures get_chroma_ingest
    adds the PDF pages to the vector store (via a Fake Chroma).
    """
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "data" / "pdfs" / "p554.pdf"
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"

    chunks = _pdf_to_page_chunks(pdf_path)
    # ensure we have at least one non-empty page for test stability
    non_empty_texts = [c["page_content"] for c in chunks if c["page_content"].strip()]
    assert non_empty_texts, "PDF has no extractable text; cannot run test reliably."

    Fake = make_fake_chroma()
    monkeypatch.setattr(chroma_vector, "Chroma", Fake)

    embeddings = object()
    chroma_vector.get_chroma_ingest(
        embeddings=embeddings,
        chunks=chunks,
        directory=tmp_path,
        collection_name="pdf_col",
    )

    inst = Fake.last_instance
    assert inst is not None
    # embedding function and persist_directory forwarded
    assert inst.embedding_function is embeddings
    assert Path(inst.persist_directory) == tmp_path

    # only pages with non-empty text should have been added
    added_texts = [d.page_content for d in inst.added_docs]
    for t in added_texts:
        assert t.strip()  # no empty-page insertions

    # confirm the count matches the non-empty pages we extracted
    assert len(inst.added_docs) == len(non_empty_texts)
    # persisted flag should be True after ingest
    assert inst.persisted is True

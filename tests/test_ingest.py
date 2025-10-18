import shutil
from pathlib import Path

import pytest

from src.ingest import ingest


@pytest.fixture
def setup_docs(tmp_path):
    # Copy a real PDF file into the temporary docs directory
    docs_dir = tmp_path / "pdfs"
    docs_dir.mkdir()
    sample_pdf_src = Path("data/pdfs/p554.pdf")
    sample_pdf_dst = docs_dir / "p554.pdf"
    shutil.copy(sample_pdf_src, sample_pdf_dst)
    persist_path = tmp_path / "chroma"
    yield docs_dir, persist_path


def test_ingest_creates_chroma_storage(setup_docs):
    docs_dir, persist_path = setup_docs
    ingest(data_dir=docs_dir, persist_path=persist_path, collection_name="test_index")
    # Check if the persist_path directory is created and not empty
    assert persist_path.exists()
    assert any(persist_path.iterdir())

#!/usr/bin/env python3
"""
Unified PDF tests
=================

Combines simple import/structure checks and integration tests.
"""

from pathlib import Path


def test_pdf_imports():
    try:
        import PyPDF2  # noqa: F401
        import pdfplumber  # noqa: F401
    except Exception as e:
        raise AssertionError(f"PDF libraries unavailable: {e}")


def test_kb_structure_exists():
    kb_dir = Path("knowledge_base")
    assert kb_dir.exists(), "knowledge_base directory should exist"
    # At least one text or PDF file
    has_files = any(kb_dir.rglob("*.txt")) or any(kb_dir.rglob("*.pdf"))
    assert has_files, "knowledge_base should contain .txt or .pdf files"


def test_sample_pdf_text_extraction():
    import pdfplumber

    kb_dir = Path("knowledge_base")
    pdf_files = list(kb_dir.glob("**/*.pdf"))
    if not pdf_files:
        return  # skip if no PDFs

    test_pdf = pdf_files[0]
    with pdfplumber.open(test_pdf) as pdf:
        assert len(pdf.pages) > 0, "PDF should have at least one page"
        text = pdf.pages[0].extract_text() or ""
        assert isinstance(text, str), "Extracted text should be a string"





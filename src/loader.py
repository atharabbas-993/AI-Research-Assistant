# loads PDF text
# Only job: read PDF → return raw text	Feeds into splitter.py

# src/loader.py

# We use pypdf's PdfReader class to read PDF files.
# Why pypdf? It's lightweight, pure Python, and handles most text-based PDFs well.
# Alternative: PyMuPDF (fitz) — faster and better at complex layouts, but heavier dependency.
from pypdf import PdfReader

# typing.List helps us clearly document what our function returns.
# This is a "type hint" — it doesn't change behavior, but tells other
# developers (and tools like VS Code) exactly what to expect.
from typing import List, Dict


def load_pdf(file_path: str) -> List[Dict]:
    """
    Reads a PDF file and extracts text from every page.

    Args:
        file_path (str): Path to the PDF file on disk.

    Returns:
        List[Dict]: A list where each item represents one page, like:
                    [{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}]

    Why return a list of dicts instead of one big string?
    Because we want to remember WHICH page each piece of text came from.
    This is needed later for citations (e.g., "Source: Page 3").
    """

    # PdfReader opens the PDF and gives us access to its pages.
    reader = PdfReader(file_path)

    # This will hold our final result — one entry per page.
    pages_data = []

    # enumerate() gives us both the index (0, 1, 2...) and the page object.
    # start=1 makes page numbers human-friendly (Page 1, not Page 0).
    for page_number, page in enumerate(reader.pages, start=1):

        # extract_text() pulls the readable text out of that page.
        # It can return None if the page has no extractable text
        # (e.g., a scanned image page) — so we guard against that.
        text = page.extract_text()

        if text:  # only keep pages that actually have text
            pages_data.append({
                "page_number": page_number,
                "text": text
            })

    return pages_data
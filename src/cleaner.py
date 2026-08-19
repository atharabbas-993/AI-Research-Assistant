# src/cleaner.py

# 're' is Python's built-in regular expressions module.
# We use it to find and replace text patterns (like extra spaces or line breaks).
import re


def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from a PDF page.

    Args:
        text (str): Raw text from one PDF page.

    Returns:
        str: Cleaned, more readable text.
    """

    # Step 1: Fix words broken by a hyphen at line-end, e.g. "meth-\nod" -> "method"
    # Pattern explanation: a hyphen, followed by a newline, followed by a lowercase letter
    text = re.sub(r'-\n(?=[a-z])', '', text)

    # Step 2: Replace all remaining newlines with a single space
    # PDFs break lines based on visual width, not sentence structure —
    # so a newline here doesn't always mean a new paragraph.
    text = re.sub(r'\n', ' ', text)

    # Step 3: Collapse multiple spaces into one
    # e.g. "Hello     world" -> "Hello world"
    text = re.sub(r'\s+', ' ', text)

    # Step 4: Remove leading/trailing whitespace
    text = text.strip()

    return text


def clean_pages(pages_data: list) -> list:
    """
    Applies clean_text() to every page in the loaded PDF data.

    Args:
        pages_data (list): Output from loader.load_pdf() —
                            list of {"page_number": int, "text": str}

    Returns:
        list: Same structure, but with cleaned text.
    """
    cleaned = []
    for page in pages_data:
        cleaned.append({
            "page_number": page["page_number"],
            "text": clean_text(page["text"])
        })
    return cleaned
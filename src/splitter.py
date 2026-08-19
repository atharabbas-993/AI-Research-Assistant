# chunking logic
# Only job: break text into chunks	Feeds into embeddings.py

# src/splitter.py

# RecursiveCharacterTextSplitter tries to split text at natural boundaries
# (paragraphs, then sentences, then words) before falling back to raw characters.
# Why "recursive"? It tries the biggest natural unit first, and only breaks
# further down if a piece is still too big.

from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List, Dict


def chunk_pages(cleaned_pages: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Splits cleaned page text into smaller overlapping chunks.

    Args:
        cleaned_pages (List[Dict]): Output from cleaner.clean_pages() —
                                     list of {"page_number": int, "text": str}
        chunk_size (int): Max characters per chunk. Default 1000.
        chunk_overlap (int): Characters repeated between consecutive chunks. Default 200.

    Returns:
        List[Dict]: Each chunk with its source page number, like:
                    [{"page_number": 1, "chunk_text": "..."}, ...]
    """

    # Create the splitter with our chosen settings.
    # separators tells it what boundaries to TRY first (in order of preference):
    # paragraph breaks, then line breaks, then sentences, then words.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    # We chunk each page SEPARATELY (not the whole document at once).
    # Why? So we can still track which page number each chunk came from —
    # important for citations later.
    for page in cleaned_pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_text in page_chunks:
            all_chunks.append({
                "page_number": page["page_number"],
                "chunk_text": chunk_text
            })

    return all_chunks
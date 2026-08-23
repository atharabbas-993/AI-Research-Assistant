from typing import List, Dict


def format_citations(sources: List[Dict]) -> str:
    """
    Formats a list of source dicts into a human-readable citation string.

    Args:
        sources (List[Dict]): list of {"source_filename", "page_number", "rerank_score"}

    Returns:
        str: A readable citation block, e.g.:
             "Sources:
              1. paper1.pdf, Page 3 (relevance: 0.87)
              2. paper1.pdf, Page 5 (relevance: 0.61)"
    """
    if not sources:
        return "No sources used."

    lines = ["Sources:"]
    for i, source in enumerate(sources, start=1):
        lines.append(
            f"{i}. {source['source_filename']}, Page {source['page_number']} "
            f"(relevance: {source['rerank_score']})"
        )

    return "\n".join(lines)
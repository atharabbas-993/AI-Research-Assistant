import json
import time
from typing import List, Dict

from src.rag_pipeline import RAGPipeline


def load_eval_dataset(path: str) -> List[Dict]:
    """Loads the test Q&A pairs from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(pipeline: RAGPipeline, eval_data: List[Dict]) -> Dict:
    """
    Checks whether the expected source/page appeared anywhere in the
    retrieved+reranked chunks, for each question that HAS an expected answer.
    """
    results = []
    correct_count = 0
    total_answerable = 0

    for item in eval_data:
        if item["expected_source_filename"] is None:
            continue

        total_answerable += 1

        retrieved = pipeline.retriever.retrieve(
            item["question"],
            top_k=10
        )

        reranked = pipeline.reranker.rerank(
            item["question"],
            retrieved,
            top_n=3
        )

        found_match = any(
            chunk["source_filename"] == item["expected_source_filename"]
            and chunk["page_number"] == item["expected_page_number"]
            for chunk in reranked
        )

        if found_match:
            correct_count += 1

        results.append({
            "question": item["question"],
            "expected": f"{item['expected_source_filename']} p{item['expected_page_number']}",
            "found_match": found_match
        })

        # Prevent Cohere rate limit
        time.sleep(7)

    accuracy = correct_count / total_answerable if total_answerable > 0 else 0

    return {
        "retrieval_accuracy": round(accuracy, 3),
        "correct": correct_count,
        "total": total_answerable,
        "details": results
    }


def evaluate_answers(pipeline: RAGPipeline, eval_data: List[Dict]) -> Dict:
    results = []
    correct_count = 0

    for item in eval_data:
        result = pipeline.ask(item["question"])

        if item["expected_answer"] is None:
            is_correct = (result["answered_from_context"] == False)
        else:
            # Support both a single string (old format) and a list of
            # acceptable keywords (new format) — pass if ANY keyword matches.
            expected = item["expected_answer"]

            if isinstance(expected, str):
                expected = [expected]

            answer_lower = result["answer"].lower()

            is_correct = any(
                kw.lower() in answer_lower
                for kw in expected
            )

        if is_correct:
            correct_count += 1

        results.append({
            "question": item["question"],
            "expected": item["expected_answer"] or "[should refuse]",
            "actual_answer": result["answer"],
            "correct": is_correct
        })

        # Prevent Cohere rate limit
        time.sleep(7)

    accuracy = correct_count / len(eval_data) if eval_data else 0

    return {
        "answer_accuracy": round(accuracy, 3),
        "correct": correct_count,
        "total": len(eval_data),
        "details": results
    }
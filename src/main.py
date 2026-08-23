import time

from rag_pipeline import RAGPipeline
from evaluator import load_eval_dataset, evaluate_retrieval, evaluate_answers

# Load your test data and initialize the pipeline once
pipeline = RAGPipeline()
eval_data = load_eval_dataset(r"D:\Workspace\AI_Research_Assistant\data\eval_dataset.json")

print(f"Loaded {len(eval_data)} test questions\n")

# --------------------------------------------
# PART 1: Retrieval Evaluation
# --------------------------------------------
print("=" * 60)
print("RETRIEVAL EVALUATION (did we find the right chunks?)")
print("=" * 60)

retrieval_results = evaluate_retrieval(pipeline, eval_data)

print(f"\nRetrieval Accuracy: {retrieval_results['retrieval_accuracy'] * 100:.1f}%")
print(f"({retrieval_results['correct']}/{retrieval_results['total']} correct)\n")

for detail in retrieval_results["details"]:
    status = "PASS" if detail["found_match"] else "FAIL"
    print(f"[{status}] {detail['question']}")
    print(f"       Expected source: {detail['expected']}")

# Wait before answer evaluation
print("\nWaiting 7 seconds before answer evaluation...")
time.sleep(7)

# --------------------------------------------
# PART 2: Answer Evaluation
# --------------------------------------------
print("\n" + "=" * 60)
print("ANSWER EVALUATION (is the final answer correct?)")
print("=" * 60)

answer_results = evaluate_answers(pipeline, eval_data)

print(f"\nAnswer Accuracy: {answer_results['answer_accuracy'] * 100:.1f}%")
print(f"({answer_results['correct']}/{answer_results['total']} correct)\n")

for detail in answer_results["details"]:
    status = "PASS" if detail["correct"] else "FAIL"
    print(f"[{status}] {detail['question']}")
    print(f"       Expected: {detail['expected']}")
    print(f"       Got: {detail['actual_answer'][:150]}")
    print()

# --------------------------------------------
# SUMMARY
# --------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Retrieval Accuracy: {retrieval_results['retrieval_accuracy'] * 100:.1f}%")
print(f"Answer Accuracy:    {answer_results['answer_accuracy'] * 100:.1f}%")
# main.py (or src/main.py, wherever you've been testing)

from retriever import Retriever
from prompt import build_prompt
from llm import LLM


# Step 1: Retrieve relevant chunks for the question
retriever = Retriever()
question = "What is Machine Learning and Deep Learning?"
retrieved_chunks = retriever.retrieve(question, top_k=3)

print(f"Retrieved {len(retrieved_chunks)} chunks\n")

# Step 2: Build the final prompt (instructions + context + question)
final_prompt = build_prompt(question, retrieved_chunks)

# Step 3: Send prompt to the LLM and get the answer
llm = LLM()
answer = llm.generate_answer(final_prompt)

# Step 4: Show the result
print("=" * 50)
print("QUESTION:", question)
print("=" * 50)
print("ANSWER:\n", answer)
print("=" * 50)

# Bonus: show which sources were used
print("\nSources used:")
for chunk in retrieved_chunks:
    print(f"- {chunk['source_filename']}, Page {chunk['page_number']} (distance: {chunk['distance']:.4f})")
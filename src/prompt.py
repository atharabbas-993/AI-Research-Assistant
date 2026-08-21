
from typing import List, Dict

# PromptTemplate is LangChain's class for building reusable prompts
# with named placeholders that get filled in at runtime.
from langchain_core.prompts import PromptTemplate


# ------------------------------------------------------------------
# Define the template ONCE, at module load time — not inside a function.
# Why? This template doesn't change between calls, so building it every
# time we generate a prompt would be wasteful. We build it once and reuse it,
# same reasoning as loading the embedding model once in Step 7.
# ------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = PromptTemplate(
    # input_variables tells LangChain exactly which placeholders
    # this template expects. LangChain will RAISE AN ERROR if you
    # forget to provide one of these — a safety net beginners often
    # don't get with plain f-strings (which fail silently or with
    # confusing KeyErrors).
    input_variables=["context", "question"],

    # template is the actual prompt text. {context} and {question}
    # are placeholders that get replaced when we call .format()
    template="""You are a helpful research assistant. Answer the user's question using ONLY the context provided below.

Rules:
- Only use information from the context below. Do not use any outside knowledge.
- If the answer is not found in the context, say "I couldn't find this information in the provided documents." Do not guess or make up an answer.
- Keep your answer clear and concise.
- When possible, mention which source/page the answer came from.

Context:
{context}

Question: {question}

Answer:"""
)


def format_context(retrieved_chunks: List[Dict]) -> str:
    """
    Formats retrieved chunks into a single readable context string,
    labeling each chunk with its source filename and page number.

    Args:
        retrieved_chunks (List[Dict]): Output from retriever.retrieve()

    Returns:
        str: Formatted context block, ready to insert into the prompt template.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['source_filename']}, Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Builds the final prompt by filling the LangChain PromptTemplate
    with the formatted context and the user's question.

    Args:
        question (str): The user's question.
        retrieved_chunks (List[Dict]): Retrieved chunks from the retriever.

    Returns:
        str: The final prompt text, ready to send to the LLM.
    """
    context_text = format_context(retrieved_chunks)

    # .format() fills in the {context} and {question} placeholders.
    # This is the LangChain-native way to produce the final prompt string —
    # equivalent to our old f-string, but validated and reusable.
    final_prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    )

    return final_prompt
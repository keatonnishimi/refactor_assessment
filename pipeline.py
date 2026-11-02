"""
This script implements a minimal Retrieval-Augmented Generation (RAG) pipeline. It retrieves relevant documents
from a predefined knowledge base using TF-IDF vectorization and cosine similarity, then generates a contextual
response to a user-provided question using a GPT-2 language model. The script evaluates the quality of the
response based on relevance, references, and length, providing a critique and confidence score. It can be run
from the command line with options to specify the number of documents to retrieve and output format.
"""

# minimal rag

import argparse
import json
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers.pipelines.base import Pipeline

### Constants
RAG_DOCS = [
    {
        "id": "1",
        "txt": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace.",
    },
    {
        "id": "2",
        "txt": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It is based on standard Python type hints and provides automatic API documentation. FastAPI is one of the fastest Python frameworks available.",
    },
    {
        "id": "3",
        "txt": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to parse data, learn from it, and make predictions or decisions without being explicitly programmed.",
    },
    {
        "id": "4",
        "txt": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate informed, contextual answers. This approach improves accuracy and reduces hallucinations.",
    },
    {
        "id": "5",
        "txt": "Vector databases store data as high-dimensional vectors and enable efficient similarity search. They are essential for modern AI applications, particularly in semantic search and retrieval systems. Common examples include Pinecone, Weaviate, and Chroma.",
    },
]
## Prompt thresholds
MIN_SIMILARITY_SCORE = 0.15

## Evaluation thresholds
MIN_ANSWER_LENGTH = 20
LOW_RELEVANCE_THRESHOLD = 0.1
VALID_SCORE_THRESHOLD = 0.6
MAX_PROBLEMS = 3

## Evaluation penalties
NO_DOCS_PENALTY = 0.4
NO_INFO_PENALTY = 0.5
SHORT_ANSWER_PENALTY = 0.3
NO_REFERENCES_PENALTY = 0.7
LOW_RELEVANCE_PENALTY = 0.6

### Refactored Functions


def parse_args():
    """Parse CLI arguments."""
    # Should be easier to maintain than sys.argv, especially if we add more args later
    # While loop was funky. argparse adds minimal overhead and gives us better arg handling
    # out of the box. Optionally, can look into typer for more functionality
    # but that may add non-trivial overhead
    parser = argparse.ArgumentParser(
        description='Ask a question! Usage: python pipeline.py "<question>" [-k NUM] [--json] [--model MODEL_NAME] [--similarity FLOAT]'
    )

    parser.add_argument(
        "question",
        help="Your question - make sure to put ' or \" around your question!",
    )
    parser.add_argument(  # Optional
        "-k",
        "--top-k",
        dest="top_k",
        type=int,
        default=3,
        help="Number of relevant docs to use, (default: 3",
    )

    parser.add_argument(  # Optional
        "--json",
        action="store_true",
        help="Print the output into json format, (default: False)",
    )

    parser.add_argument(  # Optional
        "--model", type=str, default="gpt2", help='LLM to use, try \"Qwen/Qwen2.5-0.5B-Instruct\", (default: "gpt2")'
    )

    parser.add_argument(  # Optional
        "--similarity",
        type=float,
        default=MIN_SIMILARITY_SCORE,
        help="Minimum similarity score to check against documents, (default: 0.15)",
    )

    args = parser.parse_args()

    return args


## Let's make this a function so we can handle
## Failed loads better, and configure max_new_tokens
def load_model(model_name: str = "gpt2", max_new_tokens=256) -> Pipeline:
    """Load model from HuggingFace"""
    try:
        print("Loading Model...")
        llm = pipeline(
            "text-generation", model=model_name, max_new_tokens=max_new_tokens
        )
        print("Model Loaded successfully!")
        return llm
    # ignore BLE001 blind exception "Exception".
    # This is okay for model loader to catch whatever error pipeline() could throw
    except Exception as e:  # noqa : BLE001
        print(f'Error loading model "{model_name}": {e}')
        sys.exit(1)


def retrieve_documents(
    question: str,
    docs: list[dict],
    top_k: int = 3,
    min_similarity: float = MIN_SIMILARITY_SCORE,
) -> list[dict]:
    """Get top_k documents with positive tidf cosine similarity"""
    # better name, types, default
    # check for positive k
    # use a return instead of modifying globals
    # can be used outside of linear execution now

    # make sure args exist, check top_k greater than 0
    if not docs or top_k <= 0 or not question.strip():
        return []

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform([doc["txt"] for doc in docs])
    query_vector = vectorizer.transform([question])

    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_docs = []
    for i in ranked_indices:
        if similarities[i] > min_similarity:
            retrieved_docs.append(
                {
                    "doc": docs[i],
                    "score": float(similarities[i]),
                }
            )

    return retrieved_docs


def call_model(question: str, retrieved_docs: list[dict], llm: Pipeline):
    """Generate answer from docs and question"""

    context = ""
    for counter, retrieved_doc in enumerate(retrieved_docs, start=1):
        doc_text = retrieved_doc["doc"]["txt"]
        context += f"[{counter}] {doc_text}\n\n"

    prompt = (
        f"Context:\n{context}\n"
        f"Question: {question}\n\n"
        f"Answer the question based only on the context above. Cite sources using [1], [2], etc. Be concise.\n\n"
        f"Answer:"
    )

    output = llm(prompt, return_full_text=False)
    answer = output[0]["generated_text"].strip()

    if answer == "":
        answer = "UNKNOWN: Unable to generate answer given retrieved documents. Try re-phrasing your question or increasing --top_k"

    return answer


def validate_output(
    answer: str, retrieved_docs: list[dict]
) -> tuple[bool, float, list[str]]:
    """Evaluate answer quality"""
    # Could add args here, even up to the CLI
    # For which categories we want to score on
    # low criticality though
    # Probably should add thresholds as constants
    # or configs somewhere
    problems = []
    score = 1.0

    # If no documents were retrieved, return immediately with 'no docs' penalty
    if len(retrieved_docs) == 0:
        problems.append("No documents retrieved")
        score = score * NO_DOCS_PENALTY
        return False, score, problems

    if "UNKNOWN: " in answer.upper():
        problems.append("No information in provided documents.")
        score = score * NO_INFO_PENALTY

    if len(answer) < MIN_ANSWER_LENGTH:
        problems.append("Answer was too short.")
        score = score * SHORT_ANSWER_PENALTY

    refs = re.findall(r"\[(\d+)\]", answer)

    if len(retrieved_docs) > 0 and len(refs) == 0:
        problems.append("No valid references found.")
        score = score * NO_REFERENCES_PENALTY

    # got rid of loop with sum
    avg = sum(r["score"] for r in retrieved_docs) / len(retrieved_docs)

    if avg < LOW_RELEVANCE_THRESHOLD:
        problems.append("Low Relevance")
        score = score * LOW_RELEVANCE_PENALTY

    is_valid = (score > VALID_SCORE_THRESHOLD) and (len(problems) < MAX_PROBLEMS)

    return is_valid, score, problems


def print_results(
    question: str,
    answer: str,
    retrieved_docs: list[dict],
    is_valid: bool,
    score: float,
    problems: list[str],
    use_json=False,
):
    """Display summary or json output"""
    if use_json:
        result = {
            "answer": answer,
            "citations": retrieved_docs,
            "critique": {
                "valid": is_valid,
                "score": score,
                "problems": problems,
            },
        }
        print(json.dumps(result, indent=2))
        return

    print(f"\nQ: {question}")
    print(f"\nA: {answer}\n")

    if len(retrieved_docs) == 0:
        print("Citations: None")
    else:
        print("Citations:")
        for r in retrieved_docs:
            doc_text = r["doc"]["txt"]
            if len(doc_text) > 80:
                doc_text = doc_text[:80] + "..."
            print(
                "  [{}] (score: {:.3f}) {}".format(r["doc"]["id"], r["score"], doc_text)
            )

    # can do this all at once
    validity = "OK" if is_valid else "ISSUES"
    print(f"\nCritique: {validity} (confidence: {score:.2f})")

    if len(problems) > 0:
        print(f"Problems: {', '.join(problems)}")



def main():
    args = parse_args()

    # 1. Retrieve relevant documents
    retrieved_docs = retrieve_documents(
        question=args.question,
        docs=RAG_DOCS,
        top_k=args.top_k,
        min_similarity=MIN_SIMILARITY_SCORE,
    )

    # 2. Generate answer
    if len(retrieved_docs) > 0:
        llm = load_model(model_name=args.model)
        answer = call_model(
            question=args.question, retrieved_docs=retrieved_docs, llm=llm
        )
    else:
        answer = "UNKNOWN: No documents retrieved. Try re-phrasing your question or increasing --top_k"

    # 3. Validate
    is_valid, score, problems = validate_output(
        answer=answer, retrieved_docs=retrieved_docs
    )

    # 4. Print
    print_results(
        question=args.question,
        answer=answer,
        retrieved_docs=retrieved_docs,
        is_valid=is_valid,
        score=score,
        problems=problems,
        use_json=args.json,
    )


if __name__ == "__main__":
    main()

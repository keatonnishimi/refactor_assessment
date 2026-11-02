"""
This script implements a minimal Retrieval-Augmented Generation (RAG) pipeline. It retrieves relevant documents 
from a predefined knowledge base using TF-IDF vectorization and cosine similarity, then generates a contextual 
response to a user-provided question using a GPT-2 language model. The script evaluates the quality of the 
response based on relevance, references, and length, providing a critique and confidence score. It can be run 
from the command line with options to specify the number of documents to retrieve and output format.
"""

# minimal rag

import argparse
import sys
import re
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Pipeline

def parse_args():
    """Parse CLI arguments."""
    # Should be easier to maintain than sys.argv, especially if we add more args later
    # While loop was funky. argparse adds minimal overhead and gives us better arg handling
    # out of the box. Optionally, can look into typer for more functionality 
    # but that may add non-trivial overhead
    parser = argparse.ArgumentParser(
        description="Ask a question! Usage: python pipeline.py \"<question>\" [-k NUM] [--json]"
        )
    
    parser.add_argument(
        "question", 
        help="Your question - make sure to put \' or \" around your question!"
        )
    parser.add_argument(#Optional
        "-k",
        "--top-k",
        dest="top_k",
        type=int,
        default=3,
        help="Number of relevant docs to use, (default: 3")
    
    parser.add_argument(#Optional
        "--json",
        action="store_true",
        help="Print the output into json format, (default: False)")

# These were duplicated for some reason
# Can just construct temp storage with the question
retrieved_docs = []
answer = ""
problems = []
score = 1.0
is_valid = False
temp_storage = {"question": args.question}

docs = [
    {"id": "1", "txt": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace."},
    {"id": "2", "txt": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It is based on standard Python type hints and provides automatic API documentation. FastAPI is one of the fastest Python frameworks available."},
    {"id": "3", "txt": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to parse data, learn from it, and make predictions or decisions without being explicitly programmed."},
    {"id": "4", "txt": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate informed, contextual answers. This approach improves accuracy and reduces hallucinations."},
    {"id": "5", "txt": "Vector databases store data as high-dimensional vectors and enable efficient similarity search. They are essential for modern AI applications, particularly in semantic search and retrieval systems. Common examples include Pinecone, Weaviate, and Chroma."}
]

def load_model(model_name: str="gpt2", max_new_tokens=256) -> Pipeline:
    """Load model from HuggingFace"""
    try:
        print("Loading Model...")
        llm = Pipeline("text-generation", model=model_name, max_new_tokens=max_new_tokens)
        print("Model Loaded successfully!")
        return llm
    except Exception as e:
        print(f"Error loading model \"{model_name}\": {e}")
        sys.exit(1)

def retrieve_documents(question: str, docs: list[dict], top_k: int=3) -> list[dict]:
    """Get top_k documents with positive tidf cosine similarity"""
    #better name, types, default
    #check for positive k
    #use a return instead of modifying globals
    #can be used outside of linear execution now

    #make sure args exist, check top_k greater than 0
    if not docs or top_k<=0 or not question.strip():
        return []

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform([doc["txt"] for doc in docs])
    query_vector = vectorizer.transform([question])

    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_docs = []
    for i in ranked_indices:
        if similarities[i] > 0:
            retrieved_docs.append({
                "doc": docs[i],
                "score": float(similarities[i]),
            })

    return retrieved_docs


#probably move most of this into print_results()
if len(retrieved_docs) == 0:
    answer = "idk"
    problems.append("no docs")
    score = score * 0.4
    is_valid = False
    print("\nQ: {}".format(args.question))
    print(f"\nA: {answer}\n")
    print("Citations: None")
    print("\nCritique: ISSUES (confidence: {:.2f})".format(score))
    print("Problems: %s" % ', '.join(problems))
    if args.json:
        print(json.dumps({"answer": answer, "citations": retrieved_docs, "critique": {"ok": is_valid, "score": score, "problems": problems}}, indent=2))
    sys.exit(0)


def call_model(question: str, retrieved_docs: list[dict], llm: Pipeline):
    """Generate answer from docs and question"""

    context = ""
    for counter, retrieved_doc in enumerate(retrieved_docs, start=1):
        doc_text = retrieved_doc["doc"]["txt"]
        context += f"[{counter}] {doc_text}\n\n"

    temp_storage["context"] = context

    prompt = (
        f"Context:\n{context}\n"
        f"Question: {question}\n\n"
        f"Answer the question based only on the context above. Cite sources using [1], [2], etc. Be concise.\n\n"
        f"Answer:"
    )

    output = llm(prompt, return_full_text=False)
    answer = output[0]['generated_text'].strip()

    if answer == "":
        answer = "idk"
        #need to write better fallback
        #describing failure modes here

    return answer

def validate_output():
    global problems, score, is_valid
    problems = []
    score = 1.0

    if "idk" in answer.lower():
        problems.append("no info")
        score = score * 0.5

    if len(answer) < 20:
        problems.append("short")
        score = score * 0.3

    refs = re.findall(r'\[(\d+)\]', answer)

    if len(retrieved_docs) > 0:
        if len(refs) == 0:
            problems.append("no refs")
            score = score * 0.7

        avg = 0
        for r in retrieved_docs:
            avg += r["score"]
        avg = avg / len(retrieved_docs)

        if avg < 0.1:
            problems.append("low relevance")
            score = score * 0.6

    if score > 0.6:
        if len(problems) < 3:
            is_valid = True
        else:
            is_valid = False
    else:
        is_valid = False

validate_output()


#put all this into print_results()
print("\nQ: {}".format(args.question))
print(f"\nA: {answer}\n")
print("Citations:")

for r in retrieved_docs:
    doc_text = r['doc']['txt']
    if len(doc_text) > 80:
        doc_text = doc_text[:80] + "..."
    print("  [{}] (score: {:.3f}) {}".format(r['doc']['id'], r['score'], doc_text))

if is_valid:
    print("\nCritique: OK (confidence: {:.2f})".format(score))
else:
    print(f"\nCritique: ISSUES (confidence: {score:.2f})")

if len(problems) > 0:
    print("Problems: %s" % ', '.join(problems))

result = {
    "answer": answer,
    "citations": retrieved_docs,
    "critique": {"ok": is_valid, "score": score, "problems": problems}
}

if args.json:
    print(json.dumps(result, indent=2))


def main():
    args = parse_args()

    #1. Retrieve relevant documents
    retrieved_docs = retrieve_documents(args.question, docs, top_k=args.top_k)

    #2. Generate answer
    if len(retrieved_docs) > 0:
        llm = load_model()
        answer = call_model(args.question, retrieved_docs, llm)
    else:
        answer = "idk"
        #need to write a better fallback
        #this should say something about the exception
        #that we didn't find docs relevant for whatever reason

    #3. Validate
    validate_output()

    #4. Print
    print_results()

if __name__ == "__main__":
    main()
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer


# ==============================
# Configuration
# ==============================

INDEX_FILE = "college.index"
CHUNKS_FILE = "chunks.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


# ==============================
# Load FAISS index
# ==============================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)


# ==============================
# Load chunks
# ==============================

with open(CHUNKS_FILE, "rb") as file:
    chunks = pickle.load(file)


print(f"Loaded {len(chunks)} chunks")


# ==============================
# Load embedding model
# ==============================

print("Loading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Chatbot ready!")


# ==============================
# Search knowledge base
# ==============================

def search_knowledge(question, k=5):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    distances, indices = index.search(
        question_embedding.astype("float32"),
        k
    )

    results = []

    for idx in indices[0]:

        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])

    return results


# ==============================
# Ask Ollama
# ==============================

def ask_ollama(question, context):

    prompt = f"""
You are the official Noida International University (NIU)
college information assistant.

Answer the user's question using ONLY the information
provided in the CONTEXT.

Rules:
- Give a direct and concise answer.
- Do not invent information.
- If the answer is present in the context, use it.
- For fees, mention the exact amount and academic year.
- If the information is not in the context, say:
"I don't have this information in my college knowledge base."
- Do not interpret normal questions as exit commands.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# ==============================
# Chat loop
# ==============================

print()
print("======================================")
print("       NIU College AI Chatbot")
print("======================================")
print("Type 'exit' to quit.")
print()


while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    print("\nSearching knowledge base...")

    relevant_chunks = search_knowledge(question)

    print("\n--- Retrieved Chunks ---")

    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)

    print("\n--- End Retrieved Chunks ---")

    context = "\n\n".join(relevant_chunks)

    print("\nGenerating answer...")

    try:

        answer = ask_ollama(
            question,
            context
        )

        print("\nBot:", answer)

    except requests.exceptions.ConnectionError:

        print("\nError: Ollama is not running.")

    except Exception as e:

        print(f"\nError: {e}")

    print()
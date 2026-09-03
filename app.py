import streamlit as st
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
# Page Configuration
# ==============================

st.set_page_config(
    page_title="NIU AI College Assistant",
    page_icon="🎓",
    layout="centered"
)


# ==============================
# Load Models and Data
# ==============================

@st.cache_resource
def load_resources():

    index = faiss.read_index(INDEX_FILE)

    with open(CHUNKS_FILE, "rb") as file:
        chunks = pickle.load(file)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return index, chunks, embedding_model


index, chunks, embedding_model = load_resources()


# ==============================
# Search Knowledge Base
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
# Header
# ==============================

st.title("🎓 NIU AI College Assistant")

st.write(
    "Ask questions about Noida International University "
    "courses, fees, admissions, scholarships, hostel, "
    "placements and more."
)


# ==============================
# Chat History
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==============================
# User Input
# ==============================

question = st.chat_input(
    "Ask something about NIU..."
)


if question:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # Search knowledge base
    relevant_chunks = search_knowledge(question)

    context = "\n\n".join(relevant_chunks)


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching NIU knowledge base..."):

            try:

                answer = ask_ollama(
                    question,
                    context
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except requests.exceptions.ConnectionError:

                error_message = (
                    "⚠️ Ollama is not running. "
                    "Please start Ollama and try again."
                )

                st.error(error_message)

            except Exception as e:

                st.error(f"Error: {e}")
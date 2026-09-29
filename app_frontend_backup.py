import streamlit as st
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================
INDEX_FILE = "college.index"
CHUNKS_FILE = "chunks.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

st.set_page_config(
    page_title="NIU AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROFESSIONAL NIU / CHATGPT STYLE
# =========================================================
st.markdown("""
<style>
    /* ---------- App ---------- */
    .stApp {
        background: #ffffff;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.96);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #f7faf9;
        border-right: 1px solid #e7ecea;
    }

    .brand {
        padding: 8px 8px 20px 8px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: #087f5b;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
        box-shadow: 0 5px 15px rgba(8,127,91,.15);
    }

    .brand-name {
        font-size: 22px;
        font-weight: 800;
        color: #103b31;
        line-height: 1.0;
    }

    .brand-subtitle {
        font-size: 13px;
        color: #6d817b;
        margin-top: 5px;
    }

    .sidebar-label {
        font-size: 11px;
        font-weight: 700;
        color: #75827e;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: 22px 4px 8px;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        width: 100%;
        min-height: 40px;
        border-radius: 10px;
        border: 1px solid #e0e7e4;
        background: white;
        color: #24453d;
        font-size: 14px;
        text-align: left;
        padding: 8px 12px;
        transition: all .15s ease;
    }

    div.stButton > button:hover {
        border-color: #07966b;
        background: #f0faf6;
        color: #056d50;
    }

    .new-chat div.stButton > button {
        background: #087f5b;
        border-color: #087f5b;
        color: white;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 5px 14px rgba(8,127,91,.16);
    }

    .new-chat div.stButton > button:hover {
        background: #066d4e;
        border-color: #066d4e;
        color: white;
    }

    /* ---------- Main header ---------- */
    .topbar {
        max-width: 900px;
        margin: 0 auto;
        padding: 5px 0 13px;
        border-bottom: 1px solid #eef1f0;
    }

    .topbar-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .top-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #eaf7f2;
        color: #087f5b;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
    }

    .top-title {
        font-size: 17px;
        font-weight: 750;
        color: #172b25;
    }

    .online {
        font-size: 11px;
        color: #648078;
        margin-top: 2px;
    }

    .dot {
        color: #13b87b;
        font-size: 14px;
    }

    /* ---------- Chat width ---------- */
    .main-width {
        max-width: 900px;
        margin: 0 auto;
    }

    /* ---------- Welcome ---------- */
    .welcome {
        margin: 28px auto 20px;
        padding: 22px 24px;
        border: 1px solid #dceee7;
        border-radius: 18px;
        background: linear-gradient(135deg, #f4fbf8, #fbfefd);
    }

    .welcome-row {
        display: flex;
        gap: 15px;
        align-items: flex-start;
    }

    .welcome-icon {
        min-width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #087f5b;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
    }

    .welcome-title {
        font-size: 23px;
        font-weight: 750;
        color: #173a31;
        margin: 0 0 5px;
    }

    .welcome-text {
        color: #526d65;
        font-size: 14px;
        line-height: 1.5;
        margin: 0;
    }

    /* ---------- Topic cards ---------- */
    .topic-title {
        color: #687a74;
        font-size: 12px;
        font-weight: 700;
        margin: 10px 0 7px 2px;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    /* ---------- Chat messages ---------- */
    div[data-testid="stChatMessage"] {
        padding-top: 5px !important;
        padding-bottom: 5px !important;
        gap: 10px !important;
    }

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.45;
        margin-bottom: 5px;
    }

    /* User message */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f3faf7;
        border-radius: 14px;
        margin: 5px 0;
        padding: 9px 12px !important;
    }

    /* Assistant message */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #fbfcfc;
        border: 1px solid #eef2f0;
        border-radius: 14px;
        margin: 5px 0;
        padding: 9px 12px !important;
    }

    /* ---------- Input ---------- */
    div[data-testid="stChatInput"] {
        max-width: 900px;
        margin: 14px auto 0;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
        border: 1px solid #d8e2df !important;
        background: white !important;
        min-height: 52px !important;
        box-shadow: 0 4px 18px rgba(25,60,50,.07);
    }

    div[data-testid="stChatInput"] textarea:focus {
        border-color: #087f5b !important;
        box-shadow: 0 0 0 1px #087f5b !important;
    }

    /* ---------- Footer ---------- */
    .tech {
        text-align: center;
        color: #9aa8a4;
        font-size: 10px;
        margin: 8px 0 3px;
    }

    .about {
        color: #657771;
        font-size: 12px;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD RAG
# =========================================================
@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_FILE)

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return index, chunks, embedding_model


index, chunks, embedding_model = load_resources()


# =========================================================
# SEARCH KNOWLEDGE BASE
# =========================================================
def search_knowledge(question, k=5):
    vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(vector, k)

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])

    return results


# =========================================================
# OLLAMA
# =========================================================
def ask_ollama(question, context):
    prompt = f"""
You are NIU AI Assistant, a friendly AI companion for students
of Noida International University (NIU).

You have two responsibilities:

1. COLLEGE INFORMATION
- Answer factual questions about NIU using the knowledge base below.
- This includes courses, fees, admissions, eligibility, scholarships,
  hostel, library, examinations, attendance, placements and other
  university information.
- Never invent official NIU information.
- If the requested official NIU information is not available in the
  knowledge base, say:
  "I don't have this information in my college knowledge base."

2. COLLEGE LIFE CHAT
- You can have natural conversations with students about college life.
- You can talk about classes, assignments, exams, presentations,
  studying, friends, campus life, hobbies, motivation and everyday
  student situations.
- For casual conversation, you do NOT need to force the answer to come
  from the knowledge base.
- Be friendly, natural and conversational.
- Do not sound like a formal FAQ bot.
- Keep casual responses concise and helpful.

IMPORTANT:
- Never make up NIU fees, dates, admission rules, eligibility,
  rankings, phone numbers or other official college facts.
- If a student asks for rumors, private information, unverified gossip,
  or accusations about students or staff, do not invent or spread them.
  Politely redirect them to useful or official college information.
- For official NIU questions, use the knowledge base.
- For general college-life conversation, respond naturally.

KNOWLEDGE BASE:
{context}

STUDENT QUESTION:
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


# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
        <div class="brand">
            <div class="brand-row">
                <div class="brand-icon">🎓</div>
                <div>
                    <div class="brand-name">NIU</div>
                    <div class="brand-subtitle">AI Assistant</div>
                </div>
            </div>
            <div class="brand-subtitle">Your College Information Partner</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="new-chat">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_question = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Recent Chats</div>', unsafe_allow_html=True)

    if st.session_state.recent_questions:
        for q in reversed(st.session_state.recent_questions[-7:]):
            title = q.strip()
            if len(title) > 32:
                title = title[:32] + "..."
            st.caption("💬  " + title)
    else:
        st.caption("Your recent questions will appear here.")

    st.markdown('<div class="sidebar-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="about">
        NIU AI Assistant helps students find useful information about
        courses, fees, admissions, scholarships, hostel, placements
        and other university information.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("✦  Powered by RAG + FAISS + Ollama")


# =========================================================
# TOP BAR
# =========================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-row">
        <div class="top-icon">🎓</div>
        <div>
            <div class="top-title">NIU AI Assistant</div>
            <div class="online"><span class="dot">●</span> Online</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# WELCOME SCREEN
# =========================================================
if not st.session_state.messages:
    st.markdown('<div class="main-width">', unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome">
        <div class="welcome-row">
            <div class="welcome-icon">🎓</div>
            <div>
                <div class="welcome-title">Hello! 👋</div>
                <p class="welcome-text">
                    I'm your NIU AI Assistant. Ask me about courses, fees,
                    admissions, scholarships, hostel, placements and more.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="topic-title">Quick questions</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("📄  Admission", use_container_width=True):
            st.session_state.selected_question = "How can I get admission in NIU?"
            st.rerun()

    with c2:
        if st.button("🎓  Courses", use_container_width=True):
            st.session_state.selected_question = "What courses does NIU offer?"
            st.rerun()

    with c3:
        if st.button("₹  Fees", use_container_width=True):
            st.session_state.selected_question = "What is the fee for B.Tech CSE?"
            st.rerun()

    c4, c5, c6 = st.columns(3)

    with c4:
        if st.button("🏠  Hostel", use_container_width=True):
            st.session_state.selected_question = "Tell me about hostel facilities."
            st.rerun()

    with c5:
        if st.button("💼  Placements", use_container_width=True):
            st.session_state.selected_question = "Tell me about placements at NIU."
            st.rerun()

    with c6:
        if st.button("🎓  Scholarship", use_container_width=True):
            st.session_state.selected_question = "What scholarships are available?"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# CHAT HISTORY
# =========================================================
st.markdown('<div class="main-width">', unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🎓"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# USER INPUT
# =========================================================
question = st.chat_input("Ask anything about NIU...")

if st.session_state.selected_question:
    question = st.session_state.selected_question
    st.session_state.selected_question = None


# =========================================================
# PROCESS
# =========================================================
if question:
    question = question.strip()

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        if question not in st.session_state.recent_questions:
            st.session_state.recent_questions.append(question)

        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Thinking..."):
                try:
                    relevant_chunks = search_knowledge(question)
                    context = "\n\n".join(relevant_chunks)

                    answer = ask_ollama(question, context)
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except requests.exceptions.ConnectionError:
                    answer = (
                        "I can't connect to Ollama right now. "
                        "Please make sure Ollama is running and try again."
                    )
                    st.error(answer)

                except requests.exceptions.Timeout:
                    st.error(
                        "The AI took too long to respond. Please try again."
                    )

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


if st.session_state.messages:
    st.markdown(
        '<div class="tech">NIU AI Assistant • RAG + FAISS + Ollama</div>',
        unsafe_allow_html=True
    )

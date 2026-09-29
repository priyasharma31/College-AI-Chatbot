import streamlit as st
import pickle
import faiss
import requests
import base64
from datetime import datetime
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG & CONSTANTS (UNTOUCHED BACKEND PIPELINE)
# =========================================================
INDEX_FILE = "college.index"
CHUNKS_FILE = "chunks.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

st.set_page_config(
    page_title="NIU AI - Noida International University",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# NIU BRANDING ASSETS (VECTOR SHIELD LOGO)
# =========================================================
NIU_SVG_LOGO = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%">
  <defs>
    <linearGradient id="niuGold" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#F7D070"/>
      <stop offset="50%" stop-color="#D4AF37"/>
      <stop offset="100%" stop-color="#9A7410"/>
    </linearGradient>
    <linearGradient id="niuGreen" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0F4C27"/>
      <stop offset="100%" stop-color="#072612"/>
    </linearGradient>
  </defs>
  <path d="M 50,5 L 88,20 C 88,60 70,82 50,95 C 30,82 12,60 12,20 Z" fill="url(#niuGreen)" stroke="url(#niuGold)" stroke-width="4" stroke-linejoin="round"/>
  <path d="M 50,12 L 81,24 C 81,55 66,74 50,85 C 34,74 19,55 19,24 Z" fill="none" stroke="url(#niuGold)" stroke-width="1.2" stroke-dasharray="3,2" opacity="0.7"/>
  <polygon points="50,22 68,29 50,36 32,29" fill="url(#niuGold)"/>
  <polygon points="46,35 54,35 54,40 46,40" fill="url(#niuGold)"/>
  <path d="M 34,42 Q 50,37 66,42 L 66,57 Q 50,52 34,57 Z" fill="none" stroke="url(#niuGold)" stroke-width="2"/>
  <line x1="50" y1="39" x2="50" y2="54" stroke="url(#niuGold)" stroke-width="2"/>
  <text x="50" y="74" font-family="'Inter', sans-serif" font-weight="900" font-size="16" fill="#FFFFFF" text-anchor="middle" letter-spacing="1">NIU</text>
  <text x="50" y="83" font-family="'Inter', sans-serif" font-weight="700" font-size="5.5" fill="url(#niuGold)" text-anchor="middle" letter-spacing="0.5">UNIVERSITY</text>
</svg>"""

NIU_LOGO_BASE64 = base64.b64encode(NIU_SVG_LOGO.encode('utf-8')).decode('utf-8')
NIU_LOGO_DATA_URI = f"data:image/svg+xml;base64,{NIU_LOGO_BASE64}"

# =========================================================
# CHATGPT-STYLE NIU GREEN & GOLD DESIGN SYSTEM
# =========================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Reset & Typography */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }}

    /* Hide Streamlit Chrome Header & Footer */
    #MainMenu, footer, header, [data-testid="stDecoration"], [data-testid="stHeader"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* Main Container Padding & Width Optimized for 1366x768 */
    .block-container {{
        padding-top: 0.1rem !important;
        padding-bottom: 6rem !important;
        max-width: 840px !important;
        margin: 0 auto !important;
    }}

    /* Remove excessive Streamlit vertical block gaps */
    div[data-testid="stVerticalBlock"] > div {{
        gap: 0.25rem !important;
    }}
    .element-container {{
        margin-bottom: 0px !important;
    }}

    /* Sidebar Styling - Dark Green NIU Branding */
    section[data-testid="stSidebar"] {{
        width: 260px !important;
        background-color: #0B301A !important;
        border-right: 1px solid rgba(212, 175, 55, 0.2) !important;
    }}
    
    section[data-testid="stSidebar"] > div:first-child {{
        padding: 0.85rem 0.75rem !important;
        background-color: #0B301A !important;
    }}

    .sidebar-brand-box {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 4px 12px 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 12px;
    }}

    .brand-logo-img {{
        width: 34px;
        height: 34px;
        flex-shrink: 0;
    }}

    .brand-title-text {{
        font-size: 15px;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.15;
        letter-spacing: 0.2px;
    }}

    .brand-sub-text {{
        font-size: 10.5px;
        color: #D4AF37;
        font-weight: 500;
        margin-top: 2px;
    }}

    /* New Chat Button */
    .new-chat-wrapper div.stButton > button {{
        width: 100%;
        background-color: #0F4C27 !important;
        color: #FFFFFF !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 8px !important;
        padding: 7px 12px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 8px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.15s ease !important;
    }}
    .new-chat-wrapper div.stButton > button:hover {{
        background-color: #072612 !important;
        border-color: #F7D070 !important;
        color: #F7D070 !important;
        box-shadow: 0 3px 8px rgba(212, 175, 55, 0.25) !important;
    }}

    /* Sidebar Search Input */
    div[data-testid="stSidebar"] div[data-testid="stTextInput"] input {{
        border-radius: 7px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        background: #072612 !important;
        padding: 5px 10px !important;
        font-size: 12px !important;
        color: #FFFFFF !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder {{
        color: #9CA3AF !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {{
        border-color: #D4AF37 !important;
    }}

    /* Sidebar Section Headers */
    .sidebar-section-title {{
        font-size: 10px;
        font-weight: 700;
        color: #D4AF37;
        margin: 14px 4px 6px 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* RECENT CHATS - Sleek ChatGPT Conversation History */
    .recent-btn-wrapper div.stButton > button {{
        width: 100% !important;
        background: transparent !important;
        color: #E5E7EB !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 5px 8px !important;
        font-size: 12.5px !important;
        font-weight: 400 !important;
        text-align: left !important;
        white-space: pre-wrap !important;
        line-height: 1.3 !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }}
    .recent-btn-wrapper div.stButton > button:hover {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
    }}

    /* Quick Links Items */
    .quicklink-btn-wrapper div.stButton > button {{
        width: 100% !important;
        background: transparent !important;
        color: #D1D5DB !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        text-align: left !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }}
    .quicklink-btn-wrapper div.stButton > button:hover {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
    }}

    .sidebar-divider {{
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        margin: 10px 0;
    }}

    /* Clear History Button */
    .clear-history-wrapper div.stButton > button {{
        width: 100%;
        background: transparent !important;
        color: #9CA3AF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
    }}
    .clear-history-wrapper div.stButton > button:hover {{
        color: #F87171 !important;
        border-color: rgba(248, 113, 113, 0.4) !important;
        background: rgba(239, 68, 68, 0.1) !important;
    }}

    /* Sidebar Footer Box */
    .sidebar-footer-box {{
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        font-size: 11px;
        color: #9CA3AF;
        text-align: center;
    }}
    .sidebar-footer-title {{
        font-weight: 600;
        color: #D4AF37;
        margin-bottom: 2px;
    }}

    /* MINIMAL TOP CHAT HEADER BAR */
    .top-chat-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 10px;
        background: #FFFFFF;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 8px;
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    .header-left {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .header-logo {{
        width: 24px;
        height: 24px;
    }}
    .header-title-box {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .header-title {{
        font-weight: 700;
        font-size: 14.5px;
        color: #0B301A;
    }}
    .header-status {{
        font-size: 11px;
        color: #10B981;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .status-dot {{
        width: 6.5px;
        height: 6.5px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 4px #10B981;
    }}

    /* CENTERED EMPTY STATE WELCOME SCREEN */
    .welcome-screen {{
        text-align: center;
        margin: 5vh auto 2vh auto;
        max-width: 520px;
    }}
    .welcome-logo-img {{
        width: 56px;
        height: 56px;
        margin: 0 auto 10px auto;
        filter: drop-shadow(0 3px 8px rgba(11, 48, 26, 0.12));
    }}
    .welcome-heading {{
        font-size: 24px;
        font-weight: 700;
        color: #0B301A;
        margin-bottom: 4px;
        letter-spacing: -0.3px;
    }}
    .welcome-question {{
        font-size: 17px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 4px;
    }}
    .welcome-subheading {{
        font-size: 13px;
        color: #6B7280;
        margin-bottom: 20px;
    }}

    /* SUGGESTION PILLS IN EMPTY STATE */
    .suggestion-pill-wrapper div.stButton > button {{
        width: 100%;
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 20px !important;
        padding: 6px 12px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #374151 !important;
        text-align: center !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        transition: all 0.15s ease !important;
        white-space: nowrap !important;
    }}
    .suggestion-pill-wrapper div.stButton > button:hover {{
        border-color: #0F4C27 !important;
        background-color: #E8F5E9 !important;
        color: #0B301A !important;
        box-shadow: 0 2px 6px rgba(15, 76, 39, 0.12) !important;
    }}

    /* CHAT MESSAGES STREAM STYLING */
    div[data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 4px 0 !important;
        margin: 2px 0 !important;
        min-height: auto !important;
    }}

    /* USER BUBBLE: RIGHT ALIGNED, LIGHT GREEN */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        flex-direction: row-reverse !important;
    }}
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"],
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageAvatar {{
        display: none !important;
    }}
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {{
        background-color: #E8F5E9 !important;
        border: 1px solid #C8E6C9 !important;
        padding: 9px 15px !important;
        border-radius: 18px 18px 4px 18px !important;
        color: #0B301A !important;
        max-width: 80% !important;
        margin-left: auto !important;
        font-size: 14px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }}

    /* ASSISTANT BUBBLE: LEFT ALIGNED, CLEAN BACKGROUND */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {{
        background-color: #F9FAFB !important;
        border: 1px solid #E5E7EB !important;
        padding: 11px 16px !important;
        border-radius: 4px 18px 18px 18px !important;
        color: #1F2937 !important;
        font-size: 14.5px !important;
        line-height: 1.6 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }}

    .msg-timestamp {{
        font-size: 10px;
        color: #9CA3AF;
        margin-top: 3px;
        display: block;
    }}
    .user-timestamp {{
        text-align: right;
    }}
    .assistant-timestamp {{
        text-align: left;
    }}

    /* CHAT INPUT STYLING - ChatGPT Style Docked Composer */
    div[data-testid="stChatInput"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }}
    
    div[data-testid="stChatInput"] > div {{
        border-radius: 24px !important;
        border: 1px solid #D1D5DB !important;
        background: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div[data-testid="stChatInput"] > div:focus-within {{
        border-color: #0F4C27 !important;
        box-shadow: 0 4px 20px rgba(15, 76, 39, 0.12) !important;
    }}

    div[data-testid="stChatInput"] textarea {{
        font-size: 13.5px !important;
        padding: 11px 16px !important;
        color: #111827 !important;
        background: transparent !important;
    }}

    div[data-testid="stChatInput"] button {{
        border-radius: 50% !important;
        background-color: #0F4C27 !important;
        color: #FFFFFF !important;
        border: none !important;
    }}
    div[data-testid="stChatInput"] button:hover {{
        background-color: #072612 !important;
    }}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD RAG PIPELINE (UNTOUCHED BACKEND LOGIC)
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
# SEARCH KNOWLEDGE BASE (UNTOUCHED BACKEND LOGIC)
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
# OLLAMA QA (UNTOUCHED BACKEND LOGIC)
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
# SESSION STATE INITIALIZATION
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

DEFAULT_RECENT_CHATS = [
    ("B.Tech AI & ML fee", "Today"),
    ("Admission eligibility", "Yesterday"),
    ("Hostel facilities", "2 days ago"),
    ("Scholarships details", "3 days ago"),
    ("Campus life", "5 days ago")
]

QUICK_LINKS = {
    "Admissions": "What is the admission eligibility & process for NIU?",
    "Courses & Programs": "What undergraduate and postgraduate courses does NIU offer?",
    "Fees & Financial Aid": "What is the fee structure for B.Tech CSE at NIU?",
    "Hostel & Accommodation": "Tell me about NIU hostel facilities and accommodation rules.",
    "Placements": "What are the placement opportunities and top recruiters at NIU?",
    "Campus Life": "Tell me about campus life and student activities at NIU."
}


# =========================================================
# SIDEBAR (OFFICIAL DARK GREEN BRANDED UI)
# =========================================================
with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-brand-box">
            <img class="brand-logo-img" src="{NIU_LOGO_DATA_URI}" alt="NIU Logo">
            <div>
                <div class="brand-title-text">NIU AI</div>
                <div class="brand-sub-text">Your College Assistant</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="new-chat-wrapper">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_question = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    search_term = st.text_input(
        "Search conversations",
        placeholder="Search conversations...",
        label_visibility="collapsed"
    ).strip().lower()

    # RECENT CHATS SECTION - Compact ChatGPT Style
    st.markdown('<div class="sidebar-section-title">Recent Chats</div>', unsafe_allow_html=True)

    if st.session_state.recent_questions:
        display_chats = [(q, "Today") for q in list(reversed(st.session_state.recent_questions))[-6:]]
    else:
        display_chats = DEFAULT_RECENT_CHATS

    if search_term:
        display_chats = [(q, t) for (q, t) in display_chats if search_term in q.lower()]

    if display_chats:
        for idx, (q_title, time_lbl) in enumerate(display_chats):
            btn_label = f"💬 {q_title}"
            st.markdown('<div class="recent-btn-wrapper">', unsafe_allow_html=True)
            if st.button(btn_label, key=f"rec_{idx}_{hash(q_title)}", use_container_width=True):
                st.session_state.selected_question = q_title
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.recent_questions:
            st.markdown('<div class="clear-history-wrapper" style="margin-top:6px;">', unsafe_allow_html=True)
            if st.button("Clear history", use_container_width=True):
                st.session_state.recent_questions = []
                st.session_state.messages = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("No matching chats")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # QUICK LINKS SECTION
    st.markdown('<div class="sidebar-section-title">Quick Links</div>', unsafe_allow_html=True)

    for idx, (label, prompt_text) in enumerate(QUICK_LINKS.items()):
        st.markdown('<div class="quicklink-btn-wrapper">', unsafe_allow_html=True)
        if st.button(label, key=f"ql_{idx}", use_container_width=True):
            st.session_state.selected_question = prompt_text
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # SIDEBAR FOOTER
    st.markdown("""
        <div class="sidebar-footer-box">
            <div class="sidebar-footer-title">About NIU AI</div>
            <div>Powered by RAG + FAISS + Llama 3.2</div>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# MINIMAL TOP CHAT HEADER
# =========================================================
st.markdown(f"""
    <div class="top-chat-header">
        <div class="header-left">
            <img class="header-logo" src="{NIU_LOGO_DATA_URI}" alt="NIU Logo">
            <div class="header-title-box">
                <span class="header-title">NIU AI</span>
                <span class="header-status"><span class="status-dot"></span> Online</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


# =========================================================
# MAIN CONTENT AREA: EMPTY STATE vs CHAT STREAM
# =========================================================
if not st.session_state.messages:
    # CENTERED EMPTY STATE (REMOVED WHEN CHATTING STARTS)
    st.markdown(f"""
        <div class="welcome-screen">
            <img class="welcome-logo-img" src="{NIU_LOGO_DATA_URI}" alt="NIU Badge">
            <div class="welcome-heading">NIU AI</div>
            <div class="welcome-question">How can I help you today?</div>
            <div class="welcome-subheading">Your AI assistant for Noida International University.</div>
        </div>
    """, unsafe_allow_html=True)

    # SMALL ROUNDED SUGGESTION PILLS (VISIBLE ONLY IN EMPTY STATE)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="suggestion-pill-wrapper">', unsafe_allow_html=True)
        if st.button("Admissions", use_container_width=True):
            st.session_state.selected_question = "What is the admission eligibility & process for NIU?"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="suggestion-pill-wrapper">', unsafe_allow_html=True)
        if st.button("Hostel Facilities", use_container_width=True):
            st.session_state.selected_question = "Tell me about NIU hostel facilities & rules"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="suggestion-pill-wrapper">', unsafe_allow_html=True)
        if st.button("Courses & Programs", use_container_width=True):
            st.session_state.selected_question = "What courses does NIU offer?"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="suggestion-pill-wrapper">', unsafe_allow_html=True)
        if st.button("Placement Details", use_container_width=True):
            st.session_state.selected_question = "What are the placement details and top recruiters at NIU?"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="suggestion-pill-wrapper">', unsafe_allow_html=True)
        if st.button("Fee Structure", use_container_width=True):
            st.session_state.selected_question = "What is the fee structure of B.Tech AI & ML?"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="suggestion-pill-wrapper">', unsafe_allow_html=True)
        if st.button("Scholarships", use_container_width=True):
            st.session_state.selected_question = "What scholarships are available for NIU students?"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # CONVERSATION STREAM (PURE CHAT MODE - NO PILLS OR HERO HEADINGS)
    for message in st.session_state.messages:
        role = message["role"]
        avatar = "👤" if role == "user" else NIU_LOGO_DATA_URI
        ts = message.get("timestamp", "")

        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                st.markdown(f'<div style="display:flex; align-items:center; gap:6px; font-weight:700; font-size:12.5px; color:#0B301A; margin-bottom:4px;"><img src="{NIU_LOGO_DATA_URI}" style="width:16px; height:16px;"> NIU AI</div>', unsafe_allow_html=True)
            st.markdown(message["content"])
            if ts:
                ts_class = "user-timestamp" if role == "user" else "assistant-timestamp"
                st.markdown(f'<span class="msg-timestamp {ts_class}">{ts}</span>', unsafe_allow_html=True)


# =========================================================
# CHAT INPUT COMPOSER & PROCESSING
# =========================================================
question = st.chat_input("Ask anything about NIU...")

if st.session_state.selected_question:
    question = st.session_state.selected_question
    st.session_state.selected_question = None

if question:
    question = question.strip()

    if question:
        curr_time = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": curr_time
        })

        if question not in st.session_state.recent_questions:
            st.session_state.recent_questions.append(question)

        # Trigger re-render to display user message instantly
        st.rerun()

# If last message is from user, generate response using RAG + Ollama pipeline
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_q = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant", avatar=NIU_LOGO_DATA_URI):
        st.markdown(f'<div style="display:flex; align-items:center; gap:6px; font-weight:700; font-size:12.5px; color:#0B301A; margin-bottom:4px;"><img src="{NIU_LOGO_DATA_URI}" style="width:16px; height:16px;"> NIU AI</div>', unsafe_allow_html=True)
        with st.spinner("Thinking..."):
            try:
                relevant_chunks = search_knowledge(user_q)
                context = "\n\n".join(relevant_chunks)

                answer = ask_ollama(user_q, context)
                st.markdown(answer)

                curr_time = datetime.now().strftime("%I:%M %p")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": curr_time
                })
                st.rerun()

            except requests.exceptions.ConnectionError:
                err_msg = "Unable to connect to Ollama. Please ensure the local service is running."
                st.error(err_msg)

            except requests.exceptions.Timeout:
                st.error("Response generation timed out. Please try again.")

            except Exception as e:
                st.error(f"Error: {e}")

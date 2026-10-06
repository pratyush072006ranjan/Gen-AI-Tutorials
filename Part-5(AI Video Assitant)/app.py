"""
AI Video Assistant — Streamlit UI
-----------------------------------
Wraps the existing RAG pipeline (pipeline.py) in a polished web interface.

SETUP:
1. Save your original CLI script (the one with `run_pipeline`) as `pipeline.py`
   in the same folder as this file (or edit the import below to match its
   actual filename/path).
2. Run with:  streamlit run streamlit_app.py
"""

import os
import time
import tempfile
import warnings
from datetime import datetime

import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------------------------------------------
# Import your existing pipeline. Rename/adjust this if your file is named
# differently — the function signature expected is:
#     run_pipeline(source: str, language: str = "english") -> dict
# ----------------------------------------------------------------------------
try:
    from pipeline import run_pipeline
except ImportError:
    from core.rag_engine import ask_question  # fallback so app still loads for editing
    run_pipeline = None


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — dark, glassmorphic, gradient-accented theme
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- App background ---- */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1b1030 0%, #0b0716 45%, #060409 100%);
        color: #eae6f5;
    }

    /* ---- Hide default streamlit chrome (but keep the sidebar toggle!) ---- */
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
        box-shadow: none;
    }
    /* Make the sidebar collapse/expand arrow clearly visible against dark bg */
    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="collapsedControl"] {
        background: rgba(124,58,237,0.25) !important;
        border: 1px solid rgba(196,181,253,0.4) !important;
        border-radius: 8px !important;
        color: #f5f3ff !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"] svg,
    button[data-testid="collapsedControl"] svg {
        fill: #f5f3ff !important;
    }

    /* ---- Headings ---- */
    h1, h2, h3, .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* ---- Hero header ---- */
    .hero-wrap {
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        margin-bottom: 1.6rem;
        background: linear-gradient(120deg, rgba(124,58,237,0.22), rgba(236,72,153,0.14) 55%, rgba(14,165,233,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 40px rgba(124,58,237,0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(circle at 90% 10%, rgba(236,72,153,0.25), transparent 40%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #c4b5fd, #f0abfc 45%, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: #b9b3d1;
        font-size: 1.02rem;
        margin-top: 0.4rem;
        max-width: 700px;
    }
    .badge-row { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .badge {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #d9d4ee;
    }

    /* ---- Glass cards ---- */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(196,181,253,0.35);
        transform: translateY(-2px);
    }
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        display: flex; align-items: center; gap: 0.5rem;
    }

    /* ---- Feature grid on empty state ---- */
    .feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.4rem; }
    .feat-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem;
    }
    .feat-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
    .feat-title { font-weight: 600; font-size: 0.95rem; color: #ecebfb; margin-bottom: 0.25rem; }
    .feat-desc { font-size: 0.83rem; color: #a29cc0; line-height: 1.35; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #120b22, #0a0614);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 0;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        box-shadow: 0 4px 18px rgba(124,58,237,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 22px rgba(236,72,153,0.4);
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
        color: #b9b3d1;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,58,237,0.25) !important;
        color: #f5f3ff !important;
    }

    /* ---- Chat bubbles ---- */
    .stChatMessage { border-radius: 16px; }

    /* ---- Text input / area ---- */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #eae6f5 !important;
    }

    /* ---- Divider glow ---- */
    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(196,181,253,0.5), transparent);
        margin: 1.4rem 0;
        border: none;
    }

    /* ---- Small label pills for lists ---- */
    .pill {
        display: inline-block;
        background: rgba(124,58,237,0.18);
        border: 1px solid rgba(124,58,237,0.35);
        color: #d8c7ff;
        border-radius: 8px;
        padding: 0.15rem 0.55rem;
        font-size: 0.75rem;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False


# ============================================================================
# SIDEBAR — controls
# ============================================================================
with st.sidebar:
    st.markdown("### 🎬 AI Video Assistant")
    st.caption("RAG-powered meeting & video intelligence")
    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    input_mode = st.radio("Source type", ["🔗 YouTube URL", "📁 Local / Upload file"], label_visibility="visible")

    source_value = None
    if input_mode == "🔗 YouTube URL":
        source_value = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded = st.file_uploader("Upload audio/video file", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv"])
        if uploaded is not None:
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source_value = tmp_path
            st.success(f"Ready: {uploaded.name}")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    process_clicked = st.button("✨ Process Video", disabled=st.session_state.processing)
    reset_clicked = st.button("🔄 Start Over")

    if reset_clicked:
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.processing = False
        st.rerun()

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.caption(f"🕒 {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    if st.session_state.result:
        st.caption("✅ Video processed — chat is live")


# ============================================================================
# PROCESSING
# ============================================================================
if process_clicked:
    if run_pipeline is None:
        st.error("Couldn't import `run_pipeline` from pipeline.py. Make sure your original script is saved as `pipeline.py` next to this file.")
    elif not source_value:
        st.warning("Please provide a YouTube URL or upload a file first.")
    else:
        st.session_state.processing = True
        st.session_state.chat_history = []
        progress_area = st.empty()
        stages = [
            "🎧 Extracting & chunking audio...",
            "📝 Transcribing content...",
            "🧠 Generating summary & title...",
            "✅ Pulling out action items...",
            "🔑 Identifying key decisions...",
            "❓ Spotting open questions...",
            "🔗 Building RAG knowledge base...",
        ]
        with progress_area.container():
            bar = st.progress(0, text=stages[0])
            for i, s in enumerate(stages):
                bar.progress(int(((i + 1) / len(stages)) * 90), text=s)
                time.sleep(0.15)
            try:
                result = run_pipeline(source_value, language=language)
                st.session_state.result = result
                bar.progress(100, text="🎉 Done!")
                time.sleep(0.3)
            except Exception as e:
                st.session_state.processing = False
                progress_area.empty()
                st.error(f"Something went wrong while processing: {e}")
                st.stop()
        progress_area.empty()
        st.session_state.processing = False
        st.rerun()


# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🎬 AI Video Assistant</div>
    <div class="hero-sub">Drop in a YouTube link or a local recording — get an instant summary, action items,
    key decisions, and a chat interface to ask anything about the content, powered by RAG.</div>
    <div class="badge-row">
        <span class="badge">🎙️ Transcription</span>
        <span class="badge">📋 Summarization</span>
        <span class="badge">✅ Action Items</span>
        <span class="badge">💬 RAG Chat</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# EMPTY STATE
# ============================================================================
if st.session_state.result is None and not st.session_state.processing:
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">🎧</div>
            <div class="feat-title">Any Source</div>
            <div class="feat-desc">Paste a YouTube link or upload a local audio/video file to get started.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🧠</div>
            <div class="feat-title">Smart Extraction</div>
            <div class="feat-desc">Automatic title, summary, action items, decisions, and open questions.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">💬</div>
            <div class="feat-title">Chat With It</div>
            <div class="feat-desc">Ask follow-up questions and get grounded, RAG-powered answers instantly.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Add a source in the sidebar and click **Process Video** to begin.")


# ============================================================================
# RESULTS VIEW
# ============================================================================
if st.session_state.result:
    result = st.session_state.result

    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">📌 {result.get('title', 'Untitled')}</div>
    </div>
    """, unsafe_allow_html=True)

    tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions, tab_chat = st.tabs(
        ["📋 Summary", "📝 Transcript", "✅ Action Items", "🔑 Decisions", "❓ Questions", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">📋 Summary</div>
            <div>{result.get('summary', 'No summary available.')}</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button("⬇️ Download summary", result.get("summary", ""), file_name="summary.txt")

    with tab_transcript:
        st.markdown('<div class="glass-card"><div class="card-title">📝 Full Transcript</div></div>', unsafe_allow_html=True)
        st.text_area("Transcript", result.get("transcript", ""), height=400, label_visibility="collapsed")
        st.download_button("⬇️ Download transcript", result.get("transcript", ""), file_name="transcript.txt")

    with tab_actions:
        items = result.get("action_items", "")
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">✅ Action Items</div>
            <div>{items if items else "No action items detected."}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_decisions:
        decisions = result.get("key_decisions", "")
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">🔑 Key Decisions</div>
            <div>{decisions if decisions else "No key decisions detected."}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_questions:
        questions = result.get("open_questions", "")
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">❓ Open Questions</div>
            <div>{questions if questions else "No open questions detected."}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_chat:
        st.caption("Ask anything grounded in this video's content — type your question in the box at the bottom of the page.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

        if not st.session_state.chat_history:
            st.markdown("""
            <div class="glass-card" style="text-align:center; color:#a29cc0;">
                💡 No messages yet — ask something like <i>"What were the main takeaways?"</i>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Chat input lives OUTSIDE st.tabs() intentionally.
# Streamlit's frontend has a known bug ("'setIn' cannot be called on an
# ElementNode") when st.chat_input is nested inside st.tabs + a fixed-height
# st.container — it corrupts the DOM diff and blanks the whole page.
# Keeping it at root level (pinned to the bottom of the app by Streamlit by
# default) avoids that entirely, and it still only makes sense to use while
# results exist.
# ----------------------------------------------------------------------------
if st.session_state.result:
    user_q = st.chat_input("Type your question about the video...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        try:
            from core.rag_engine import ask_question
            with st.spinner("Thinking..."):
                answer = ask_question(st.session_state.result["rag_chain"], user_q)
        except Exception as e:
            answer = f"⚠️ Error while answering: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()
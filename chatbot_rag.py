#import streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# vector store
from langchain_community.vectorstores import FAISS

# import langchain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import google.generativeai as genai
from pathlib import Path

# load .env from project root deterministically
env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parents[0] / ".env"
load_dotenv(dotenv_path=str(env_path), override=True)
# also read raw dotenv values to ensure creds are available
from dotenv import dotenv_values
raw_vals = dotenv_values(str(env_path)) if env_path.exists() else {}
if raw_vals.get("ADMIN_USER"):
    os.environ["ADMIN_USER"] = (raw_vals["ADMIN_USER"] or "").strip().strip('"').strip("'")
if raw_vals.get("ADMIN_PASSWORD"):
    os.environ["ADMIN_PASSWORD"] = (raw_vals["ADMIN_PASSWORD"] or "").strip().strip('"').strip("'")

# startup diagnostics and normalization
print("[startup] dotenv:", str(env_path))
raw_env_model = os.environ.get("GEMINI_CHAT_MODEL", "")
# normalize: ensure fully-qualified name starts with 'models/'
normalized_env_model = raw_env_model.strip().strip('"').strip("'")
if normalized_env_model and not normalized_env_model.startswith("models/"):
    normalized_env_model = f"models/{normalized_env_model}"
os.environ["GEMINI_CHAT_MODEL"] = normalized_env_model
print("[startup] GEMINI_CHAT_MODEL:", os.environ.get("GEMINI_CHAT_MODEL"))

st.title("AI Assistant")

# single-page admin login gate
source = "env"
try:
    import streamlit as _stref
    if hasattr(_stref, "secrets") and _stref.secrets:
        admin_user = (_stref.secrets.get("ADMIN_USER", "") or "").strip().strip('"').strip("'")
        admin_pass = (_stref.secrets.get("ADMIN_PASSWORD", "") or "").strip().strip('"').strip("'")
        source = "secrets"
    else:
        admin_user = (os.environ.get("ADMIN_USER", "") or "").strip().strip('"').strip("'")
        admin_pass = (os.environ.get("ADMIN_PASSWORD", "") or "").strip().strip('"').strip("'")
except Exception:
    admin_user = (os.environ.get("ADMIN_USER", "") or "").strip().strip('"').strip("'")
    admin_pass = (os.environ.get("ADMIN_PASSWORD", "") or "").strip().strip('"').strip("'")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.form("admin_login"):
        u = st.text_input("Username").strip()
        p = st.text_input("Password", type="password").strip()
        submitted = st.form_submit_button("Login")
    if submitted:
        if admin_user and admin_pass and u == admin_user and p == admin_pass:
            st.session_state.authenticated = True
            try:
                st.rerun()
            except Exception:
                try:
                    import streamlit as _st_legacy
                    _st_legacy.experimental_rerun()
                except Exception:
                    pass
            st.stop()
        else:
            st.error("Invalid credentials")
    st.stop()

# get admin creds (prefer Streamlit secrets if available), else env; normalize
source = "env"
try:
    import streamlit as _st_ref
    if hasattr(_st_ref, "secrets") and _st_ref.secrets:  # prefer secrets first
        admin_user = (_st_ref.secrets.get("ADMIN_USER", "") or "").strip().strip('"').strip("'")
        admin_pass = (_st_ref.secrets.get("ADMIN_PASSWORD", "") or "").strip().strip('"').strip("'")
        source = "secrets"
    else:
        admin_user = (os.environ.get("ADMIN_USER", "") or "").strip().strip('"').strip("'")
        admin_pass = (os.environ.get("ADMIN_PASSWORD", "") or "").strip().strip('"').strip("'")
except Exception:
    admin_user = (os.environ.get("ADMIN_USER", "") or "").strip().strip('"').strip("'")
    admin_pass = (os.environ.get("ADMIN_PASSWORD", "") or "").strip().strip('"').strip("'")

# diagnostics
try:
    import streamlit as _st_dbg
    sec_keys = list(_st_dbg.secrets.keys()) if hasattr(_st_dbg, "secrets") else []
    print("[startup] secrets keys:", sec_keys)
except Exception:
    pass
print(f"[startup] ADMIN_PASSWORD set: {bool(admin_pass)} (source={source})")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
# logout control (shown only after successful login)
if st.session_state.get("authenticated"):
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.pop("messages", None)
        try:
            st.rerun()
        except Exception:
            pass

# initialize embeddings + local FAISS vector store (Gemini)
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    transport="rest",
)
vector_store = FAISS.load_local(
    "faiss_gemini",
    embeddings,
    allow_dangerous_deserialization=True,
)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("How are you?")

# did the user submit a prompt?
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # strict: require GEMINI_CHAT_MODEL from .env and force usage for this session
    model_env = os.environ.get("GEMINI_CHAT_MODEL")
    if not model_env:
        st.error("Set GEMINI_CHAT_MODEL in .env to a supported model, e.g., models/gemini-2.5-flash")
        st.stop()
    selected_model = model_env
    # if a previous session model exists but differs from current .env, clear it
    if st.session_state.get("gemini_model") and st.session_state["gemini_model"] != selected_model:
        del st.session_state["gemini_model"]
    st.session_state["gemini_model"] = selected_model
    print(f"[chat] using model: {selected_model}")

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        transport="rest",
    )

    # retrieval
    docs_text = ""  # ensure defined even if retrieval fails before assignment
    try:
        print("[chat] invoking retriever...")
        print("[chat] embedding query...")
        qvec = embeddings.embed_query(prompt)
        print(f"[chat] embed_ok dim={len(qvec)}")
        print("[chat] querying FAISS by vector...")
        docs = vector_store.similarity_search_by_vector(qvec, k=2)
        print(f"[chat] retrieved {len(docs)} docs")
        docs_text = "".join(d.page_content for d in docs)
    except Exception as e:
        import traceback
        print("[chat] retrieval error:", e)
        traceback.print_exc()
        st.error(f"Retrieval error: {e}")
        st.stop()

    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Context: {context}:"""

    # truncate context to reduce token usage
    context_max = 2000
    system_prompt_fmt = system_prompt.format(context=docs_text[:context_max])

    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    # Build a clean message history for Gemini: single SystemMessage first, then prior human/ai turns
    clean_messages = [SystemMessage(system_prompt_fmt)]
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            # skip any existing system messages to avoid multiple system roles
            continue
        clean_messages.append(msg)

    # attempt invoke with fallback over stable models if a 404 NotFound occurs
    candidates = []
    if os.environ.get("GEMINI_CHAT_MODEL"):
        candidates = [selected_model]
    else:
        candidates = [selected_model]
        extras = [
            "models/gemini-1.5-flash", "gemini-1.5-flash",
            "models/gemini-1.0-pro", "gemini-1.0-pro",
            "models/gemini-pro", "gemini-pro",
        ]
        for m in extras:
            if m not in candidates and "preview" not in m and "-exp" not in m:
                candidates.append(m)

    result = None
    last_err = None
    for m in candidates:
        try:
            print(f"[chat] invoking LLM with model: {m} ...")
            llm = ChatGoogleGenerativeAI(
                model=m,
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                transport="rest",
            )
            # Invoke with sanitized message list (single leading SystemMessage)
            result = llm.invoke(clean_messages).content
            print("[chat] LLM response received")
            # cache working model for this session
            st.session_state["gemini_model"] = m
            break
        except Exception as e:
            import traceback
            msg = str(e)
            print("[chat] LLM error with model", m, ":", msg)
            traceback.print_exc()
            last_err = e
            if "NotFound" in msg or "404" in msg:
                # try next candidate
                continue
            if "TooManyRequests" in msg or "quota" in msg.lower():
                st.error("Gemini quota exceeded. Add billing or switch to a model with available quota (set GEMINI_CHAT_MODEL in .env), then restart.")
                st.stop()
            st.error(f"LLM error: {e}")
            st.stop()

    if result is None:
        st.error(f"All candidate models failed. Last error: {last_err}")
        st.stop()

    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))

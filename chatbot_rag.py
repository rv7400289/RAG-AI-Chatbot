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
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# Paths
documents_dir = Path(__file__).resolve().parent / "documents"
documents_dir.mkdir(exist_ok=True)
index_dir = Path(__file__).resolve().parent / "faiss_gemini"

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    transport="rest",
)

# Build/rebuild index helper
def build_index():
    loader = PyPDFDirectoryLoader(str(documents_dir))
    raw_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )
    documents = splitter.split_documents(raw_documents)
    vs = FAISS.from_documents(documents=documents, embedding=embeddings)
    vs.save_local(str(index_dir))
    st.session_state["vector_store"] = vs

# Load index if present
if "vector_store" not in st.session_state:
    try:
        st.session_state["vector_store"] = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        st.session_state["vector_store"] = None

# Sidebar: upload + reindex
uploaded = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded:
    saved = 0
    for f in uploaded:
        dest = documents_dir / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        saved += 1
    st.sidebar.success(f"Uploaded {saved} file(s) to documents/")

if st.sidebar.button("Update Index"):
    try:
        build_index()
        st.sidebar.success("Index updated")
    except Exception as e:
        st.sidebar.error(f"Index update failed: {e}")

# Sidebar: optional document filter for retrieval
available_pdfs = sorted([p.name for p in documents_dir.glob("*.pdf")])
st.session_state.setdefault("doc_filter", [])
st.session_state["doc_filter"] = st.sidebar.multiselect(
    "Limit answers to selected document(s)", options=available_pdfs, default=st.session_state.get("doc_filter", [])
)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks. "))

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

    if st.session_state["vector_store"] is None:
        st.error("Vector index not found. Upload PDFs and click 'Update Index' first.")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        transport="rest",
    )

    try:
        print("[chat] invoking retriever...")
        print("[chat] embedding query...")
        qvec = embeddings.embed_query(prompt)
        print(f"[chat] embed_ok dim={len(qvec)}")
        print("[chat] querying FAISS by vector...")
        # pull more candidates to allow post-filtering by filename
        docs = st.session_state["vector_store"].similarity_search_by_vector(qvec, k=20)
        print(f"[chat] retrieved {len(docs)} docs (pre-filter)")
        # optional filter by selected filenames in sidebar
        selected_files = set(st.session_state.get("doc_filter", []) or [])
        if selected_files:
            def _is_selected(doc):
                src = (doc.metadata.get("source") or "").strip()
                return Path(src).name in selected_files
            docs = [d for d in docs if _is_selected(d)]
            print(f"[chat] {len(docs)} docs after filter by {list(selected_files)}")
        if not docs:
            st.warning("No matching chunks found in the selected document(s). Try selecting different docs or clearing the filter.")
            st.stop()
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

    # Single SystemMessage first, then prior human/ai turns
    clean_messages = [SystemMessage(system_prompt_fmt)]
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue
        clean_messages.append(msg)

    # Model candidates (mostly redundant because we enforce env model)
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

#import streamlit
import streamlit as st
import os
import base64
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pathlib import Path

# Load .env deterministically
env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parents[0] / ".env"
load_dotenv(dotenv_path=str(env_path), override=True)
from dotenv import dotenv_values
raw_vals = dotenv_values(str(env_path)) if env_path.exists() else {}
if raw_vals.get("ADMIN_USER"):
    os.environ["ADMIN_USER"] = (raw_vals["ADMIN_USER"] or "").strip().strip('"').strip("'")
if raw_vals.get("ADMIN_PASSWORD"):
    os.environ["ADMIN_PASSWORD"] = (raw_vals["ADMIN_PASSWORD"] or "").strip().strip('"').strip("'")

# Startup diagnostics and normalization
print("[startup] dotenv:", str(env_path))
raw_env_model = os.environ.get("GEMINI_CHAT_MODEL", "")
normalized_env_model = raw_env_model.strip().strip('"').strip("'")
if normalized_env_model and not normalized_env_model.startswith("models/"):
    normalized_env_model = f"models/{normalized_env_model}"
os.environ["GEMINI_CHAT_MODEL"] = normalized_env_model
print("[startup] GEMINI_CHAT_MODEL:", os.environ.get("GEMINI_CHAT_MODEL"))

st.title("AI Assistant")

# Admin login gate
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

# Diagnostics
try:
    import streamlit as _st_dbg
    sec_keys = list(_st_dbg.secrets.keys()) if hasattr(_st_dbg, "secrets") else []
    print("[startup] secrets keys:", sec_keys)
except Exception:
    pass
print(f"[startup] ADMIN_PASSWORD set: {bool(admin_pass)} (source={source})")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
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

# GitHub helpers
def _gh_headers():
    token = st.secrets.get("GITHUB_TOKEN") if hasattr(st, "secrets") else os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set in secrets.")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

def _gh_base():
    owner = st.secrets.get("GITHUB_OWNER") if hasattr(st, "secrets") else os.environ.get("GITHUB_OWNER")
    repo = st.secrets.get("GITHUB_REPO") if hasattr(st, "secrets") else os.environ.get("GITHUB_REPO")
    branch = st.secrets.get("GITHUB_BRANCH") if hasattr(st, "secrets") else os.environ.get("GITHUB_BRANCH", "main")
    if not owner or not repo:
        raise RuntimeError("GITHUB_OWNER or GITHUB_REPO missing in secrets.")
    return owner, repo, branch

def gh_get_file_sha(dest_path: str):
    owner, repo, branch = _gh_base()
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dest_path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def gh_put_file(local_path: str, dest_path: str, message_prefix: str = "chore: upload"):
    owner, repo, branch = _gh_base()
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dest_path}"
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")
    sha = gh_get_file_sha(dest_path)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = {
        "message": f"{message_prefix} {dest_path} at {ts}",
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=data, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub commit failed for {dest_path}: {r.status_code} {r.text}")
    return r.json()

def gh_put_directory(local_dir: Path, dest_dir: str, message_prefix: str = "chore: upload dir"):
    uploaded = 0
    for root, _, files in os.walk(local_dir):
        for name in files:
            local_path = Path(root) / name
            rel = local_path.relative_to(local_dir)
            dest_path = f"{dest_dir}/{rel.as_posix()}"
            gh_put_file(str(local_path), dest_path, message_prefix=message_prefix)
            uploaded += 1
    return uploaded

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

# Sidebar: upload, reindex, push
uploaded = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
push_to_github = st.sidebar.checkbox("Also push uploads to GitHub", value=False)
push_index_btn = st.sidebar.button("Push index folder to GitHub")

if uploaded:
    saved = 0
    for f in uploaded:
        dest = documents_dir / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        saved += 1
        if push_to_github:
            try:
                gh_put_file(str(dest), f"documents/{f.name}", message_prefix="feat: add document")
            except Exception as e:
                st.sidebar.error(f"GitHub upload failed for {f.name}: {e}")
    st.sidebar.success(f"Uploaded {saved} file(s) to documents/")

if st.sidebar.button("Update Index"):
    try:
        build_index()
        st.sidebar.success("Index updated")
        if push_to_github or push_index_btn:
            try:
                count = gh_put_directory(index_dir, "faiss_gemini", message_prefix="feat: update index")
                st.sidebar.success(f"Pushed {count} index files to GitHub")
            except Exception as e:
                st.sidebar.error(f"GitHub index push failed: {e}")
    except Exception as e:
        st.sidebar.error(f"Index update failed: {e}")

# Sidebar: optional document filter for retrieval
available_pdfs = sorted([p.name for p in documents_dir.glob("*.pdf")])
st.session_state.setdefault("doc_filter", [])
st.session_state["doc_filter"] = st.sidebar.multiselect(
    "Limit answers to selected document(s)", options=available_pdfs, default=st.session_state.get("doc_filter", [])
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("How are you?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    model_env = os.environ.get("GEMINI_CHAT_MODEL")
    if not model_env:
        st.error("Set GEMINI_CHAT_MODEL in .env to a supported model, e.g., models/gemini-2.5-flash")
        st.stop()
    selected_model = model_env
    if st.session_state.get("gemini_model") and st.session_state["gemini_model"] != selected_model:
        del st.session_state["gemini_model"]
    st.session_state["gemini_model"] = selected_model
    print(f"[chat] using model: {selected_model}")

    if st.session_state["vector_store"] is None:
        st.error("Vector index not found. Upload PDFs and click 'Update Index' first.")
        st.stop()

    # Retrieval
    docs_text = ""
    try:
        print("[chat] invoking retriever...")
        print("[chat] embedding query...")
        qvec = embeddings.embed_query(prompt)
        print(f"[chat] embed_ok dim={len(qvec)}")
        print("[chat] querying FAISS by vector...")
        docs = st.session_state["vector_store"].similarity_search_by_vector(qvec, k=20)
        print(f"[chat] retrieved {len(docs)} docs (pre-filter)")
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

    context_max = 2000
    system_prompt_fmt = system_prompt.format(context=docs_text[:context_max])

    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    clean_messages = [SystemMessage(system_prompt_fmt)]
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue
        clean_messages.append(msg)

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
            st.session_state["gemini_model"] = m
            break
        except Exception as e:
            import traceback
            msg = str(e)
            print("[chat] LLM error with model", m, ":", msg)
            traceback.print_exc()
            last_err = e
            if "NotFound" in msg or "404" in msg:
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

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from rag_local.app_core import (
    answer_turn,
    cfg_with_overrides,
    get_index_chunk_count,
    init_index,
    refresh_index,
    run_conversation_eval_from_messages,
    get_corpus_docs_dir,
    get_corpus_index_path,
    get_corpus_root,
    delete_current_index,
    delete_current_corpus,
    default_corpus_id,
    list_corpora,
)
from rag_local.eval_rag import format_eval_report
from rag_local.openalex_client import OpenAlexClient
from rag_local.openalex_fetch import materialize_openalex_selected
from rag_local.paper_rerank import SearchProfile, rerank_layered
from rag_local.wiki_client import search_wikipedia
from rag_local.wiki_fetch import materialize_wikipedia_selected

from rag_local.chat_store import save_chat, load_chat, derive_chat_title, get_chat_dir, list_chats

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
SEARCH_STRATEGY_PRESETS = {
    "Balanced": {
        "semantic": 0.50,
        "bm25": 0.20,
        "lexical": 0.15,
        "llm": 0.10,
        "audience": 0.05,
        "prefilter_k": 25,
    },
    "Academic Precision": {
        "semantic": 0.35,
        "bm25": 0.35,
        "lexical": 0.20,
        "llm": 0.05,
        "audience": 0.05,
        "prefilter_k": 20,
    },
    "Broad Overview": {
        "semantic": 0.55,
        "bm25": 0.10,
        "lexical": 0.15,
        "llm": 0.10,
        "audience": 0.10,
        "prefilter_k": 30,
    },
    "Beginner-Friendly": {
        "semantic": 0.40,
        "bm25": 0.10,
        "lexical": 0.15,
        "llm": 0.10,
        "audience": 0.25,
        "prefilter_k": 25,
    },
    "Research-Heavy": {
        "semantic": 0.45,
        "bm25": 0.25,
        "lexical": 0.20,
        "llm": 0.05,
        "audience": 0.05,
        "prefilter_k": 20,
    },
}

def resolve_search_strategy(
    strategy_name: str,
    *,
    advanced_override: bool,
    semantic: float,
    bm25: float,
    lexical: float,
    llm: float,
    audience: float,
    prefilter_k: int,
):
    preset = SEARCH_STRATEGY_PRESETS.get(strategy_name, SEARCH_STRATEGY_PRESETS["Balanced"])

    if not advanced_override:
        return (
            preset["semantic"],
            preset["bm25"],
            preset["lexical"],
            preset["llm"],
            preset["audience"],
            preset["prefilter_k"],
        )

    return semantic, bm25, lexical, llm, audience, prefilter_k

# -----------------------------
# Cached OpenAlex search (API)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_openalex_search(query: str, limit: int, mailto: str):
    client = OpenAlexClient()
    mailto_val = mailto.strip() or None
    return client.search_works(
        query=query,
        limit=int(limit),
        mailto=mailto_val,
        open_access_only=True,
    )

def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_option" not in st.session_state:
        st.session_state.model_option = "llama3.1:8b" 
    if "index" not in st.session_state:
        st.session_state.index = None
    if "cfg" not in st.session_state:
        st.session_state.cfg = {}
    if "eval_result" not in st.session_state:
        st.session_state.eval_result = None
    if "corpus_id" not in st.session_state:
        st.session_state.corpus_id = default_corpus_id()

    if "chat_id" not in st.session_state:
        st.session_state.chat_id = st.session_state.corpus_id

    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "search_selected" not in st.session_state:
        st.session_state.search_selected = {}
    if "last_search_mode" not in st.session_state:
        st.session_state.last_search_mode = "Academic (OpenAlex)"
    if "show_search_results" not in st.session_state:
        st.session_state.show_search_results = False
    if "search_result_summary" not in st.session_state:
        st.session_state.search_result_summary = ""
    if "last_fetch_saved" not in st.session_state:
        st.session_state.last_fetch_saved = []
    if "last_fetch_skipped" not in st.session_state:
        st.session_state.last_fetch_skipped = []

_init_state()

st.sidebar.markdown("---")
st.sidebar.subheader("Corpus")

_base_data_dir = Path(st.session_state.cfg.get("data_dir", "rag_local/Data")).resolve()
existing_corpora = list_corpora(st.session_state.cfg)

_saved_chats = {}  # corpus_id -> title
for _cid in existing_corpora:
    _root = _base_data_dir / "corpora" / _cid
    _record = load_chat(_root, _cid)
    if _record and _record.messages:
        _index_exists = (_root / ".index" / "local_index.json").exists()
        _status = "indexed" if _index_exists else "no index"
        _saved_chats[_cid] = f"{_record.title} - {_status}"

if _saved_chats:
    # build display labels (label with status) -> corpus_id lookup
    _chat_labels = {label: cid for cid, label in _saved_chats.items()}

    _is_new_unsaved = st.session_state.corpus_id not in _saved_chats
    _current_label = _saved_chats.get(st.session_state.corpus_id, None)
    _placeholder = "— New chat —"
    _label_options = ([_placeholder] if _is_new_unsaved else []) + list(_chat_labels.keys())

    selected_label = st.sidebar.selectbox(
        "Switch chat",
        options=_label_options,
        index=0 if _is_new_unsaved else (_label_options.index(_current_label) if _current_label in _label_options else 0),
        help="Select a saved chat to resume it.",
    )

    selected_corpus = _chat_labels.get(selected_label, st.session_state.corpus_id)

    _delete_btn_disabled = selected_label == _placeholder
    if st.sidebar.button("Delete selected chat", key="delete_selected_chat_btn", use_container_width=True, disabled=_delete_btn_disabled):
        _del_root = _base_data_dir / "corpora" / selected_corpus
        import shutil as _shutil
        if _del_root.exists():
            _shutil.rmtree(_del_root)
        if selected_corpus == st.session_state.corpus_id:
            st.session_state.corpus_id = default_corpus_id()
            st.session_state.chat_id = st.session_state.corpus_id
            st.session_state.index = None
            st.session_state.messages = []
        st.rerun()

    # rename chat section
    if selected_label != _placeholder:
        _current_title = _saved_chats.get(selected_corpus, "").rsplit(" - ", 1)[0]
        _rename_input = st.sidebar.text_input("Rename chat", value=_current_title, key="rename_input")
        if st.sidebar.button("Save name", key="save_name_btn", use_container_width=True):
            _rename_root = _base_data_dir / "corpora" / selected_corpus
            _rename_record = load_chat(_rename_root, selected_corpus)
            if _rename_record:
                save_chat(_rename_root, chat_id=selected_corpus, messages=_rename_record.messages, title=_rename_input.strip())
            st.rerun()

    # only switch if the current corpus_id is actually in the existing list (not a brand new one)
    if selected_corpus != st.session_state.corpus_id and st.session_state.corpus_id in existing_corpora:
        if st.session_state.messages:
            _cur_root = get_corpus_root(st.session_state.cfg)
            save_chat(_cur_root, chat_id=st.session_state.chat_id, messages=st.session_state.messages)
        st.session_state.corpus_id = selected_corpus
        st.session_state.chat_id = selected_corpus
        st.session_state.cfg = cfg_with_overrides(st.session_state.cfg, corpus_id=selected_corpus)
        st.session_state.search_results = []
        st.session_state.search_selected = {}
        st.session_state.show_search_results = False
        st.session_state.last_fetch_saved = []
        st.session_state.last_fetch_skipped = []
        _new_root = _base_data_dir / "corpora" / selected_corpus
        _record = load_chat(_new_root, selected_corpus)
        st.session_state.messages = _record.messages if _record else []
        # auto-load index for the switched corpus if it exists
        try:
            _switched_cfg = cfg_with_overrides(st.session_state.cfg, corpus_id=selected_corpus)
            _idx, _meta = init_index(_switched_cfg, force_rebuild=False)
            st.session_state.index = _idx
        except Exception:
            st.session_state.index = None
        st.rerun()

# corpus_id driven entirely by session state; no text input to avoid overwrite conflicts
corpus_id = st.session_state.corpus_id

new_corpus_btn = st.sidebar.button("New corpus", use_container_width=True)

if new_corpus_btn:

    if st.session_state.messages:
        _cur_root = get_corpus_root(st.session_state.cfg)
        save_chat(_cur_root, chat_id=st.session_state.chat_id, messages=st.session_state.messages)
    _new_id = default_corpus_id()
    st.session_state.corpus_id = _new_id
    st.session_state.chat_id = _new_id

    st.session_state.cfg = cfg_with_overrides(st.session_state.cfg, corpus_id=_new_id)
    st.session_state.index = None
    st.session_state.messages = []
    st.session_state.eval_result = None
    st.session_state.search_results = []
    st.session_state.search_selected = {}
    st.session_state.show_search_results = False
    st.session_state.last_fetch_saved = []
    st.session_state.last_fetch_skipped = []
    st.rerun()

effective_docs_dir = get_corpus_docs_dir(st.session_state.cfg)
effective_index_path = get_corpus_index_path(st.session_state.cfg)

st.sidebar.caption(f"Docs: {effective_docs_dir}")
st.sidebar.caption(f"Index: {effective_index_path}")

# -----------------------------
# Sidebar: settings
# -----------------------------
st.sidebar.title("Settings")

modify_ollama_settings = st.sidebar.checkbox(
    "Modify Ollama Model Settings",
    value=False,
    key="modify_ollama_settings",
    help="Enable Ollama setting controls",
)

ollama_host = st.sidebar.text_input(
    "Ollama host",
    value=st.session_state.cfg.get("ollama_host", "http://localhost:11434"),
    disabled=not modify_ollama_settings
)
#chat_model = st.sidebar.text_input(
#    "Chat model",
#    value=st.session_state.cfg.get("chat_model", st.session_state.cfg.get("model", "llama3.1:8b")),
#)

chat_model = st.sidebar.selectbox(
    "Chat model",
    options=["llama3.1:8b", "llama3.2:3b"],
    index=0 if st.session_state.cfg.get("model") == "llama3.2:3b" else 0,
    help="Select the active Ollama model. 3.1:8b is default for precision; 3.2:3b is faster for lighter hardware."
)

system_prompt = st.sidebar.text_area(
    "System prompt",
    value=st.session_state.cfg.get(
        "system_prompt",
        """You are a RAG study assistant.

Core behavior:
- Use the provided SOURCES as the primary ground truth.
- If the answer is not in the sources, say “Not found in the provided sources” and suggest what to search for.
- Explain step-by-step, but keep it tight and structured.
- When you make a claim supported by sources, cite it inline like [S1], [S2].
- Prefer definitions, then intuition, then a worked example when helpful.

Be strict about citations when RAG is enabled."""
    ),
    height=140,
    disabled=not modify_ollama_settings
)

st.sidebar.markdown("---")
rag_enabled = st.sidebar.toggle(
    "Enable RAG",
    value=bool(st.session_state.cfg.get("rag_enabled", True)),
)

top_k = st.sidebar.slider(
    "top_k",
    min_value=1,
    max_value=15,
    value=int(st.session_state.cfg.get("top_k", 5)),
)
max_context_chars = st.sidebar.slider(
    "max_context_chars",
    min_value=2000,
    max_value=20000,
    value=int(st.session_state.cfg.get("max_context_chars", 6000)),
    step=500,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Index")

data_dir = st.sidebar.text_input(
    "Data dir",
    value=st.session_state.cfg.get("data_dir", "rag_local/Data"),
)
index_path = st.sidebar.text_input(
    "Index path",
    value=st.session_state.cfg.get("index_path", "rag_local/Data/.index/local_index.json"),
)
chunk_size = st.sidebar.number_input(
    "chunk_size",
    min_value=200,
    max_value=3000,
    value=int(st.session_state.cfg.get("chunk_size", 800)),
    step=50,
)
overlap = st.sidebar.number_input(
    "overlap",
    min_value=0,
    max_value=1000,
    value=int(st.session_state.cfg.get("overlap", 200)),
    step=25,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Embeddings")

embed_backend = st.sidebar.selectbox(
    "embed_backend",
    options=["ollama", "sbert"],
    index=0 if st.session_state.cfg.get("embed_backend", "ollama") == "ollama" else 1,
)
embed_model = st.sidebar.text_input(
    "embed_model",
    value=st.session_state.cfg.get("embed_model", "nomic-embed-text"),
    key=f"embed_model_input_{st.session_state.corpus_id}",
)

st.sidebar.markdown("---")
colA, colB = st.sidebar.columns(2)
load_idx = colA.button("Load index", key="load_idx_btn", use_container_width=True)
rebuild_idx = colB.button("Rebuild", key="rebuild_idx_btn", use_container_width=True)

colC, colD = st.sidebar.columns(2)
delete_idx_btn = colC.button("Delete index", key="delete_idx_btn", use_container_width=True)
delete_corpus_btn = colD.button("Delete corpus", key="delete_corpus_btn", use_container_width=True)

clear_chat = st.sidebar.button("Clear chat", key="clear_chat_btn", use_container_width=True)

st.session_state.cfg = cfg_with_overrides(
    st.session_state.cfg,
    ollama_host=ollama_host,
    chat_model=chat_model,
    model=chat_model,
    system_prompt=system_prompt,
    rag_enabled=rag_enabled,
    top_k=top_k,
    max_context_chars=max_context_chars,
    data_dir=data_dir,
    index_path=index_path,
    chunk_size=chunk_size,
    overlap=overlap,
    embed_backend=embed_backend,
    embed_model=embed_model,
    use_corpus_mode=True,
    corpus_id=st.session_state.corpus_id,
)

if clear_chat:
    st.session_state.messages = []

if load_idx:
    try:
        idx, meta = init_index(st.session_state.cfg, force_rebuild=False)
        st.session_state.index = idx
        st.sidebar.success(f"Index loaded. chunks={meta.get('chunk_count', get_index_chunk_count(idx))}")
    except Exception as e:
        st.sidebar.error(f"Load index failed: {e}")

if rebuild_idx:
    try:
        st.cache_data.clear()
        st.cache_resource.clear()

        progress_placeholder = st.sidebar.empty()
        status_placeholder = st.sidebar.empty()

        progress_bar = progress_placeholder.progress(0, text="Starting rebuild...")

        def ui_progress(frac: float, message: str) -> None:
            pct = int(max(0.0, min(1.0, frac)) * 100)
            progress_bar.progress(pct, text=message)
            status_placeholder.caption(f"{pct}% · {message}")

        idx, meta = refresh_index(
            st.session_state.cfg,
            force_rebuild=True,
            progress_callback=ui_progress,
        )
        st.session_state.index = idx

        progress_bar.progress(100, text="Rebuild complete.")
        status_placeholder.caption("100% · Rebuild complete.")

        st.sidebar.success(
            f"Index rebuilt. docs={meta.get('doc_count', 'n/a')} chunks={meta.get('chunk_count', get_index_chunk_count(idx))}"
        )

    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {e}")
if delete_idx_btn:
    try:
        deleted = delete_current_index(st.session_state.cfg)
        st.session_state.index = None
        if deleted:
            st.sidebar.success("Current local_index.json deleted.")
        else:
            st.sidebar.info("No current local_index.json found.")
    except Exception as e:
        st.sidebar.error(f"Delete index failed: {e}")

if delete_corpus_btn:
    try:
        deleted = delete_current_corpus(st.session_state.cfg)
        st.session_state.index = None
        st.session_state.messages = []
        st.session_state.search_results = []
        st.session_state.search_selected = {}
        st.session_state.show_search_results = False
        st.session_state.last_fetch_saved = []
        st.session_state.last_fetch_skipped = []

        if deleted:
            st.sidebar.success("Current corpus deleted.")
        else:
            st.sidebar.info("No current corpus directory found.")
    except Exception as e:
        st.sidebar.error(f"Delete corpus failed: {e}")

# -----------------------------
# Sidebar: evaluation
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Evaluation")

run_eval_btn = st.sidebar.button("Evaluate current conversation", use_container_width=True)

if run_eval_btn:
    assistant_turns = [m for m in st.session_state.messages if m.get("role") == "assistant"]
    if not assistant_turns:
        st.sidebar.warning("No assistant turns in the current chat yet.")
    else:
        try:
            with st.sidebar.spinner("Evaluating current conversation..."):
                st.session_state.eval_result = run_conversation_eval_from_messages(
                    st.session_state.messages
                )
            st.sidebar.success("Conversation evaluation complete.")
        except Exception as e:
            st.sidebar.error(f"Evaluation failed: {e}")

# -----------------------------
# Sidebar: Search (Academic / Wikipedia / Hybrid)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Search")

search_mode = st.sidebar.selectbox(
    "Search Mode",
    options=["Academic (OpenAlex)", "General Knowledge (Wikipedia)", "Hybrid"],
    index=["Academic (OpenAlex)", "General Knowledge (Wikipedia)", "Hybrid"].index(
        st.session_state.get("last_search_mode", "Academic (OpenAlex)")
    ),
    key="search_mode",
)
st.session_state.last_search_mode = search_mode

search_query = st.sidebar.text_input(
    "Keywords or phrase",
    key="search_query",
    value=st.session_state.get("search_query", ""),
    help=(
        "Use this for focused terms or exact phrases.\n\n"
        "Best when you already know target concepts.\n\n"
        "Examples:\n"
        "- photosynthesis in plants\n"
        "- CRISPR ethics\n"
        "- transformer attention\n\n"
        "Required with Broad topic: fill at least one of the two.\n"
        "Fallback: if Broad topic is empty, search uses this value."
    ),
)

search_topic = st.sidebar.text_input(
    "Broad topic",
    key="search_topic",
    value=st.session_state.get("search_topic", ""),
    help=(
        "Use this for broader subject context when exploring.\n\n"
        "Examples:\n"
        "- cell biology\n"
        "- climate change adaptation\n"
        "- large language models\n\n"
        "Required with Keywords: fill at least one of the two.\n"
        "Fallback: if Keywords is empty, this becomes both the raw search query "
        "and reranking topic."
    ),
)

search_research_question = st.sidebar.text_area(
    "Specific question to answer",
    key="search_research_question",
    value=st.session_state.get("search_research_question", ""),
    height=90,
    help=(
        "Optional but strongly recommended.\n\n"
        "Use this for the exact question you want answered, compared, or explained.\n"
        "It improves reranking toward answer usefulness.\n\n"
        "Examples:\n"
        "- How does amoeba energy metabolism differ from tree photosynthesis?\n"
        "- Which methods improve retrieval quality in small RAG datasets?"
    ),
)

search_level = st.sidebar.selectbox(
    "Education / familiarity level",
    options=["high_school", "undergraduate", "masters", "phd"],
    index=1,
    key="search_level",
)

oa_mailto = st.sidebar.text_input(
    "OpenAlex mailto (recommended)",
    key="oa_mailto",
    value=st.session_state.get("oa_mailto", ""),
)

if search_mode == "Academic (OpenAlex)":
    search_limit_default = 50
elif search_mode == "General Knowledge (Wikipedia)":
    search_limit_default = 20
else:
    search_limit_default = 30

search_limit = st.sidebar.slider(
    "Initial candidate pool",
    10, 200, search_limit_default, step=5, key="search_limit"
)

strategy_name = st.sidebar.selectbox(
    "Search strategy",
    options=[
        "Balanced",
        "Academic Precision",
        "Broad Overview",
        "Beginner-Friendly",
        "Research-Heavy",
    ],
    index=0,
    key="strategy_name",
    help="Recommended preset for layered reranking. Use Advanced controls only if you want manual tuning.",
)

with st.sidebar.expander("Advanced weight controls", expanded=False):

    advanced_override = st.checkbox(
        "Advanced controls",
        value=False,
        key="advanced_override",
        help="Enable manual reranking controls.",
    )

    # Manual defaults
    search_prefilter_k_manual = st.slider(
        "Lexical prefilter keep",
        5, 100, 25, step=5, key="search_prefilter_k_manual",
        disabled=not advanced_override,
    )

    search_w_sem_manual = st.slider(
        "Semantic weight",
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        key="search_w_sem_manual",
        disabled=not advanced_override,
    )
    search_w_bm25_manual = st.slider(
        "BM25 weight",
        min_value=0.0, max_value=1.0, value=0.20, step=0.05,
        key="search_w_bm25_manual",
        disabled=not advanced_override,
    )
    search_w_lex_manual = st.slider(
        "Lexical intent weight",
        min_value=0.0, max_value=1.0, value=0.15, step=0.05,
        key="search_w_lex_manual",
        disabled=not advanced_override,
    )
    search_w_llm_manual = st.slider(
        "LLM judge weight",
        min_value=0.0, max_value=1.0, value=0.10, step=0.05,
        key="search_w_llm_manual",
        disabled=not advanced_override,
    )
    search_w_audience_manual = st.slider(
        "Audience-level weight",
        min_value=0.0, max_value=1.0, value=0.05, step=0.05,
        key="search_w_audience_manual",
        disabled=not advanced_override,
    )

    use_llm_reranker = st.checkbox(
        "Enable LLM reranker (optional)",
        value=False,
        key="search_use_llm_reranker",
        help="Uses Ollama generation for the last scoring layer. Safe fallback if unavailable.",
    )

    search_llm_model = st.text_input(
        "LLM reranker model",
        value=st.session_state.get("search_llm_model", "llama3.1:8b"),
        key="search_llm_model",
        help="Only used if LLM reranker is enabled.",
    )

submitted_search = st.sidebar.button("Search", key="search_submit_btn", use_container_width=True)

search_w_sem, search_w_bm25, search_w_lex, search_w_llm, search_w_audience, search_prefilter_k = resolve_search_strategy(
    strategy_name,
    advanced_override=advanced_override,
    semantic=search_w_sem_manual,
    bm25=search_w_bm25_manual,
    lexical=search_w_lex_manual,
    llm=search_w_llm_manual,
    audience=search_w_audience_manual,
    prefilter_k=search_prefilter_k_manual,
)

if submitted_search:
    if not search_query.strip() and not search_topic.strip():
        st.sidebar.warning("Enter at least a search query or research topic.")
    else:
        try:
            normalized_candidates: List[Dict[str, Any]] = []
            total_openalex = 0
            total_wikipedia = 0
            openalex_with_pdf = 0

            raw_query = search_query.strip() or search_topic.strip()

            if search_mode in {"Academic (OpenAlex)", "Hybrid"}:
                with st.sidebar.spinner("Searching OpenAlex..."):
                    works = cached_openalex_search(
                        query=raw_query,
                        limit=int(search_limit),
                        mailto=oa_mailto,
                    )

                total_openalex = len(works)

                for w in works:
                    candidate = {
                        "id": w.id,
                        "title": w.title,
                        "abstract": w.abstract,
                        "year": w.year,
                        "url": w.url,
                        "pdf_url": w.pdf_url,
                        "source": "openalex",
                        "raw": {
                            "source": "openalex",
                            "openalex_id": w.id,
                            "landing_url": w.url,
                            "pdf_url": w.pdf_url,
                        },
                    }
                    normalized_candidates.append(candidate)
                    if str(w.pdf_url or "").strip():
                        openalex_with_pdf += 1

            if search_mode in {"General Knowledge (Wikipedia)", "Hybrid"}:
                wiki_limit = int(search_limit) if search_mode != "Hybrid" else max(10, int(search_limit) // 2)

                with st.sidebar.spinner("Searching Wikipedia..."):
                    wiki_results = search_wikipedia(raw_query, limit=wiki_limit)

                total_wikipedia = len(wiki_results)
                normalized_candidates.extend(wiki_results)

            profile = SearchProfile(
                user_query=search_query.strip(),
                topic=search_topic.strip() or search_query.strip(),
                research_question=search_research_question.strip(),
                level=search_level,
            )

            ranked, warnings = rerank_layered(
                profile=profile,
                candidates=normalized_candidates,
                ollama_host=st.session_state.cfg.get("ollama_host", "http://localhost:11434"),
                embed_model=st.session_state.cfg.get("embed_model", "nomic-embed-text"),
                llm_model=search_llm_model.strip() or "llama3.1:8b",
                use_llm_reranker=bool(use_llm_reranker),
                lexical_keep_k=int(search_prefilter_k),
                top_n=10,
                alpha_semantic=float(search_w_sem),
                beta_bm25=float(search_w_bm25),
                gamma_lexical=float(search_w_lex),
                delta_llm=float(search_w_llm),
                epsilon_audience=float(search_w_audience),
            )

            st.session_state.search_results = ranked
            st.session_state.search_selected = {}
            st.session_state.show_search_results = True

            if search_mode == "Academic (OpenAlex)":
                st.session_state.search_result_summary = (
                    f"OpenAlex returned {total_openalex} candidates. "
                    f"{openalex_with_pdf} had direct PDFs. Showing top {len(ranked)}."
                )
            elif search_mode == "General Knowledge (Wikipedia)":
                st.session_state.search_result_summary = (
                    f"Wikipedia returned {total_wikipedia} candidates. "
                    f"Showing top {len(ranked)}."
                )
            else:
                st.session_state.search_result_summary = (
                    f"Hybrid search returned {total_openalex} academic + {total_wikipedia} Wikipedia candidates. "
                    f"Showing top {len(ranked)}."
                )

            st.sidebar.success(st.session_state.search_result_summary)

            for warn in warnings:
                st.sidebar.warning(warn)

        except Exception as e:
            st.sidebar.error(f"Search failed: {e}")

if st.session_state.get("search_result_summary"):
    st.sidebar.caption(st.session_state.search_result_summary)

ranked_results = st.session_state.get("search_results", [])
if ranked_results and st.session_state.get("show_search_results", False):
    # Keep this open by default so checkbox interactions don't feel like the
    # list "disappeared" on rerun.
    with st.sidebar.expander("Candidate sources", expanded=True):
        for r in ranked_results:
            key = f"pick_{r.source}_{r.id}"
            year_part = f" ({r.year})" if r.year else ""
            label = f"[{r.source}] {r.score:.3f} | {r.title}{year_part}"

            picked = st.checkbox(
                label,
                value=bool(st.session_state.search_selected.get(key, False)),
                key=key,
            )
            st.session_state.search_selected[key] = picked

        fetch_btn = st.button("Fetch selected", key="fetch_btn", use_container_width=True)

        if fetch_btn:
            chosen = [
                r for r in ranked_results
                if st.session_state.search_selected.get(f"pick_{r.source}_{r.id}", False)
            ]

            if not chosen:
                st.warning("Select at least one result.")
            else:
                oa_payloads: List[Dict[str, Any]] = []
                wiki_payloads: List[Dict[str, Any]] = []

                for r in chosen:
                    if r.source == "openalex":
                        oa_payloads.append(
                            {
                                "openalex_id": r.id,
                                "title": r.title,
                                "abstract": r.abstract,
                                "year": r.year,
                                "url": r.url,
                                "pdf_url": r.pdf_url,
                                "score": r.score,
                                "raw": {
                                    "source": "openalex",
                                    "openalex_id": r.id,
                                    "landing_url": r.url,
                                    "pdf_url": r.pdf_url,
                                    "score": r.score,
                                    "semantic_score": r.semantic_score,
                                    "lexical_score": r.lexical_score,
                                    "bm25_score": r.bm25_score,
                                    "llm_score": r.llm_score,
                                    "audience_score": r.audience_score,
                                    "filter_notes": r.filter_notes,
                                    "research_topic": search_topic.strip(),
                                    "research_question": search_research_question.strip(),
                                    "education_level": search_level,
                                },
                            }
                        )
                    elif r.source == "wikipedia":
                        wiki_payloads.append(
                            {
                                "id": r.id,
                                "title": r.title,
                                "url": r.url,
                                "abstract": r.abstract,
                                "score": r.score,
                                "raw": {
                                    "source": "wikipedia",
                                    "score": r.score,
                                    "semantic_score": r.semantic_score,
                                    "lexical_score": r.lexical_score,
                                    "bm25_score": r.bm25_score,
                                    "llm_score": r.llm_score,
                                    "audience_score": r.audience_score,
                                    "filter_notes": r.filter_notes,
                                    "research_topic": search_topic.strip(),
                                    "research_question": search_research_question.strip(),
                                    "education_level": search_level,
                                },
                            }
                        )

                try:
                    total_saved = []
                    total_skipped = []

                    with st.spinner("Fetching selected sources..."):
                        if oa_payloads:
                            oa_saved, oa_skipped = materialize_openalex_selected(
                                selected=oa_payloads,
                                data_dir=get_corpus_docs_dir(st.session_state.cfg),
                                query=(search_research_question.strip() or search_topic.strip() or search_query.strip()),
                            )
                            total_saved.extend(oa_saved)
                            total_skipped.extend(oa_skipped)

                        if wiki_payloads:
                            wiki_saved, wiki_skipped = materialize_wikipedia_selected(
                                selected=wiki_payloads,
                                data_dir=get_corpus_docs_dir(st.session_state.cfg),
                                query=(search_research_question.strip() or search_topic.strip() or search_query.strip()),
                            )
                            total_saved.extend(wiki_saved)
                            total_skipped.extend(wiki_skipped)

                        st.session_state.last_fetch_saved = total_saved
                        st.session_state.last_fetch_skipped = total_skipped

                    if total_saved:
                        oa_fulltext = [s for s in total_saved if s.get("source") == "openalex" and s.get("content_tier") == "fulltext"]
                        oa_metadata = [s for s in total_saved if s.get("source") == "openalex" and s.get("content_tier") == "metadata_only"]
                        wiki_saved_count = len([s for s in total_saved if s.get("source") == "wikipedia"])

                        st.success(
                            f"Saved {len(total_saved)} source(s): "
                            f"{len(oa_fulltext)} academic full-text, "
                            f"{len(oa_metadata)} academic metadata-only, "
                            f"{wiki_saved_count} Wikipedia articles."
                        )

                    if total_skipped:
                        preview = "\n".join([f"- {pid}: {reason}" for pid, reason in total_skipped[:6]])
                        st.warning(f"Skipped {len(total_skipped)}:\n{preview}")

                    # Collapse and clear picker after fetch
                    st.session_state.show_search_results = False
                    st.session_state.search_selected = {}

                except Exception as e:
                    st.error(f"Fetch failed: {e}")
# -----------------------------
# Main: chat UI
# -----------------------------
st.title("Local RAG Chatbot")
st.caption("Streamlit UI · OpenAlex fetch wired into local index · GUI eval enabled")

if st.session_state.index is None:
    try:
        idx, _meta = init_index(st.session_state.cfg, force_rebuild=False)
        st.session_state.index = idx
    except Exception:
        st.session_state.index = None

# Evaluation panel
eval_result = st.session_state.get("eval_result")
if eval_result:
    with st.expander("Conversation evaluation", expanded=False):
        summary = eval_result.get("summary", {})
        rows = eval_result.get("rows", [])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Turns", summary.get("n", 0))
        c2.metric("Relevance", f"{summary.get('relevance_avg', 0.0):.2f}")
        c3.metric("Groundedness", f"{summary.get('groundedness_avg', 0.0):.2f}")
        c4.metric("Retrieval rel.", f"{summary.get('retrieval_relevance_avg', 0.0):.2f}")
        c5.metric("Citation cov.", f"{summary.get('citation_coverage_avg', 0.0):.2f}")

        st.markdown(f"**Average latency:** {summary.get('latency_avg_s', 0.0):.2f}s")
        if rows:
            display_rows = []
            for r in rows:
                display_rows.append(
                    {
                        "turn_index": r.get("turn_index"),
                        "question": r.get("question"),
                        "relevance": r.get("relevance"),
                        "groundedness": r.get("groundedness"),
                        "retrieval_relevance": r.get("retrieval_relevance"),
                        "citation_coverage": r.get("citation_coverage"),
                        "latency_s": r.get("latency_s"),
                    }
                )
            st.dataframe(display_rows, use_container_width=True)

        st.code(format_eval_report(eval_result))

# Last fetch report
last_saved = st.session_state.get("last_fetch_saved", [])
last_skipped = st.session_state.get("last_fetch_skipped", [])

if last_saved or last_skipped:
    with st.expander("Last fetch report", expanded=False):
        oa_fulltext = [s for s in last_saved if s.get("source") == "openalex" and s.get("content_tier") == "fulltext"]
        oa_metadata = [s for s in last_saved if s.get("source") == "openalex" and s.get("content_tier") == "metadata_only"]
        wiki_saved = [s for s in last_saved if s.get("source") == "wikipedia"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Academic full-text", len(oa_fulltext))
        c2.metric("Academic metadata-only", len(oa_metadata))
        c3.metric("Wikipedia saved", len(wiki_saved))
        c4.metric("Skipped", len(last_skipped))

        if oa_fulltext:
            st.markdown("**Academic full-text sources**")
            for s in oa_fulltext:
                st.markdown(f"- {s.get('title','unknown')} ({s.get('year','n/a')})")

        if oa_metadata:
            st.markdown("**Academic metadata-only fallback**")
            for s in oa_metadata:
                st.markdown(f"- {s.get('title','unknown')} ({s.get('year','n/a')}): {s.get('note','')}")

        if wiki_saved:
            st.markdown("**Wikipedia articles**")
            for s in wiki_saved:
                st.markdown(f"- {s.get('title','unknown')}")
# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("Sources", expanded=False):
                for i, s in enumerate(m["sources"], start=1):
                    st.markdown(f"**[S{i}]** `{s.get('file','')}`")
                    st.caption(f"chunk_id={s.get('chunk_id','')} · score={s.get('score','')}")
                    snippet = s.get("snippet", "")
                    if snippet:
                        st.code(snippet[:800])

prompt = st.chat_input("Ask a question")
if prompt:
    st.session_state.show_search_results = False
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            import time

            t0 = time.perf_counter()
            try:
                reply, sources = answer_turn(
                    history=st.session_state.messages,
                    user_text=prompt,
                    cfg=st.session_state.cfg,
                    index=st.session_state.index,
                )
            except Exception as e:
                reply, sources = f"Error: {e}", []
            latency_s = time.perf_counter() - t0

        st.markdown(reply)

        if sources:
            with st.expander("Sources", expanded=False):
                for i, s in enumerate(sources, start=1):
                    st.markdown(f"**[S{i}]** `{s.get('file','')}`")
                    st.caption(f"chunk_id={s.get('chunk_id','')} · score={s.get('score','')}")
                    snippet = s.get("snippet", "")
                    if snippet:
                        st.code(snippet[:800])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "sources": sources,
            "latency_s": round(latency_s, 4),
        }
    )

    _auto_save_root = get_corpus_root(st.session_state.cfg)
    save_chat(_auto_save_root, chat_id=st.session_state.chat_id, messages=st.session_state.messages)

# Search explanation panel
ranked_results = st.session_state.get("search_results", [])
if ranked_results:
    with st.expander("Search result explanations", expanded=False):
        st.markdown(
            "These are the sources that survived filtering and reranking. "
            "Use this to inspect why each result was selected."
        )

        for idx, r in enumerate(ranked_results, start=1):
            title_line = f"{idx}. [{r.source}] {r.title}"
            if r.year:
                title_line += f" ({r.year})"

            with st.container():
                st.markdown(f"### {title_line}")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Final", f"{r.score:.3f}")
                c2.metric("Semantic", f"{r.semantic_score:.2f}")
                c3.metric("Lexical", f"{r.lexical_score:.2f}")
                c4.metric("BM25", f"{r.bm25_score:.2f}")
                c5.metric("Audience", f"{r.audience_score:.2f}")

                c6, c7 = st.columns(2)
                with c6:
                    st.markdown(f"**Source lane:** {r.source}")
                    st.markdown(f"**Inferred level:** {r.inferred_level}")
                    st.markdown(f"**Has PDF:** {'Yes' if r.pdf_url else 'No'}")
                    if r.url:
                        st.markdown(f"**URL:** {r.url}")
                    if r.pdf_url:
                        st.markdown(f"**PDF:** {r.pdf_url}")

                with c7:
                    st.markdown("**Why it survived**")
                    notes = list(r.filter_notes or [])
                    if r.source == "openalex" and r.pdf_url:
                        notes = ["direct PDF available"] + notes
                    if r.source == "wikipedia":
                        notes = ["general-knowledge source"] + notes
                    if notes:
                        for note in notes:
                            st.markdown(f"- {note}")
                    else:
                        st.markdown("- passed hygiene and layered ranking")

                topic_hits = r.matched_topic_terms or []
                rq_hits = r.matched_question_terms or []

                col_topic, col_rq = st.columns(2)
                with col_topic:
                    st.markdown("**Matched topic terms**")
                    if topic_hits:
                        st.markdown(", ".join(topic_hits))
                    else:
                        st.markdown("_No strong topic-term matches found._")

                with col_rq:
                    st.markdown("**Matched research-question terms**")
                    if rq_hits:
                        st.markdown(", ".join(rq_hits))
                    else:
                        st.markdown("_No strong research-question-term matches found._")

                with st.expander("Summary / abstract preview", expanded=False):
                    preview = (r.abstract or "").strip()
                    if preview:
                        st.write(preview[:2500])
                    else:
                        st.write("No summary available.")

                st.divider()

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import settings
from src.evaluation import run_evaluation, run_red_team
from src.rag_pipeline import EnterprisePolicyAssistant
from src.vector_store import rebuild_vector_store


st.set_page_config(
    page_title="Sentinel | Policy Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        :root {
            --bg-canvas: #f8fafc;
            --bg-card: #ffffff;
            --border-subtle: #e2e8f0;
            --border-hover: #cbd5e1;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --brand-blue: #2563eb;
            --brand-blue-hover: #1d4ed8;
            --brand-blue-subtle: #eff6ff;
            --badge-green-bg: #ecfdf5;
            --badge-green-text: #059669;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: var(--text-primary);
        }

        .stApp {
            background-color: var(--bg-canvas);
        }

        /* Container & Layout */
        .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        /* Header Bar */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem 1.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }
        .header-logo {
            font-size: 1.75rem;
            display: grid;
            place-items: center;
            width: 46px;
            height: 46px;
            background: var(--brand-blue-subtle);
            border-radius: var(--radius-md);
            border: 1px solid #bfdbfe;
        }
        .header-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.2;
        }
        .header-subtitle {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
        }
        .header-badges {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: #f1f5f9;
            color: var(--text-secondary);
            border: 1px solid var(--border-subtle);
        }
        .badge-pill.status-green {
            background: var(--badge-green-bg);
            color: var(--badge-green-text);
            border-color: #a7f3d0;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] * {
            color: #f1f5f9;
        }
        [data-testid="stSidebar"] hr {
            border-color: #1e293b;
            margin: 1.1rem 0;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 1.25rem;
        }
        .sidebar-brand-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.02em;
        }
        .sidebar-brand-badge {
            font-size: 0.7rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .sidebar-policy-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.65rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            color: #e2e8f0;
            margin-bottom: 0.35rem;
        }

        /* Empty State & Suggestions */
        .empty-welcome {
            text-align: center;
            padding: 2.2rem 1.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            margin: 1rem 0 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }
        .empty-welcome h3 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.4rem;
        }
        .empty-welcome p {
            color: var(--text-secondary);
            font-size: 0.9rem;
            max-width: 600px;
            margin: 0 auto 1.25rem;
            line-height: 1.5;
        }

        /* Chat Messages */
        .user-bubble-row {
            display: flex;
            justify-content: flex-end;
            margin: 1rem 0;
        }
        .user-bubble {
            max-width: min(78%, 720px);
            padding: 0.85rem 1.1rem;
            border-radius: 16px 16px 4px 16px;
            background: var(--brand-blue);
            color: #ffffff !important;
            font-size: 0.95rem;
            line-height: 1.55;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
            word-wrap: break-word;
        }
        [data-testid="stChatMessage"] {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1rem 1.25rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
            color: var(--text-primary) !important;
            line-height: 1.6;
        }

        /* Chat Input */
        [data-testid="stChatInput"] {
            border-radius: var(--radius-md);
            border: 1px solid #cbd5e1;
            background: #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        [data-testid="stChatInput"] textarea {
            color: var(--text-primary) !important;
            font-size: 0.95rem;
        }

        /* Evidence Source Cards */
        .source-box {
            padding: 0.75rem 0.95rem;
            background: #f8fafc;
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--brand-blue);
            border-radius: var(--radius-sm);
            margin: 0.45rem 0;
            font-size: 0.85rem;
        }
        .source-box-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.2rem;
        }
        .source-box-excerpt {
            color: var(--text-muted);
            line-height: 1.5;
        }

        /* Trust & Verification Status Bar */
        .trust-badge-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.75rem;
            padding-top: 0.65rem;
            border-top: 1px solid var(--border-subtle);
            font-size: 0.78rem;
            color: var(--text-muted);
            flex-wrap: wrap;
        }
        .trust-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.2rem 0.55rem;
            border-radius: 9999px;
            background: #f1f5f9;
            color: var(--text-secondary);
            font-weight: 500;
        }
        .trust-pill.verified {
            background: #ecfdf5;
            color: #047857;
        }

        /* Buttons & Tabs */
        .stButton>button {
            border-radius: var(--radius-sm);
            font-weight: 500;
            font-size: 0.88rem;
            transition: all 0.15s ease;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: var(--bg-card);
            padding: 0.3rem;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            margin-bottom: 1.25rem;
        }
        button[data-baseweb="tab"] {
            border-radius: var(--radius-sm);
            font-weight: 600;
            font-size: 0.88rem;
            padding: 0.55rem 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _document_count() -> int:
    extensions = {".pdf", ".docx", ".txt", ".md"}
    if not settings.policy_dir.exists():
        return 0
    return sum(
        1
        for path in settings.policy_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📎 Supporting Evidence ({len(sources)} sources cited)", expanded=False):
        for source in sources:
            details = []
            if source.page:
                details.append(f"Page {source.page}")
            if source.section:
                details.append(escape(source.section))
            suffix = f" · {' · '.join(details)}" if details else ""
            st.markdown(
                f'<div class="source-box">'
                f'<div class="source-box-title">[{escape(source.id)}] {escape(source.source)}{suffix}</div>'
                f'<div class="source-box-excerpt">{escape(source.excerpt)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_trust_bar(result) -> None:
    score_text = f"{result.groundedness_score:.0%}" if result.groundedness_score is not None else "Grounded"
    is_safe = not result.blocked
    pii_status = "PII Sanitized" if result.pii_redacted else "Clean"
    
    st.markdown(
        f"""
        <div class="trust-badge-row">
            <span class="trust-pill verified">🛡️ {score_text}</span>
            <span class="trust-pill">✓ {'Allowed' if is_safe else 'Blocked'}</span>
            <span class="trust-pill">🔒 {pii_status}</span>
            <span class="trust-pill">📄 {len(result.sources)} Sources</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _render_user_message(content: str) -> None:
    safe_content = escape(content).replace("\n", "<br>")
    st.markdown(
        f'<div class="user-bubble-row"><div class="user-bubble">{safe_content}</div></div>',
        unsafe_allow_html=True,
    )


# Apply styles
_load_styles()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------------------------------------------------
# Sidebar Navigation & Settings
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div style="font-size:1.5rem;">🛡️</div>
            <div>
                <div class="sidebar-brand-title">Sentinel</div>
                <div class="sidebar-brand-badge">Policy Copilot</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Approved Policies")
    policy_files = sorted([f.name for f in settings.policy_dir.glob("*.md")])
    for pf in policy_files:
        clean_name = pf.replace("_", " ").replace(".md", "").title()
        st.markdown(
            f'<div class="sidebar-policy-item"><span>📄</span> {clean_name}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("##### Inference Engine")
    groq_models = [
        "openai/gpt-oss-120b (GPT-120)",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "Custom model…",
    ]
    active_groq = settings.groq_model
    default_idx = 0
    for i, m in enumerate(groq_models):
        if active_groq in m:
            default_idx = i
            break

    chosen_groq = st.selectbox("Model", groq_models, index=default_idx, label_visibility="collapsed")
    if chosen_groq == "Custom model…":
        selected_model = st.text_input("Model ID", value="openai/gpt-oss-120b")
    else:
        selected_model = chosen_groq.split(" ")[0]

    selected_provider = "groq"

    st.divider()
    if st.button("🗑️  Clear Conversation", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

    with st.expander("⚙️ Admin Indexer", expanded=False):
        st.caption("Re-sync vectors with `data/policies/`.")
        if st.button("↻ Rebuild Index", use_container_width=True):
            try:
                with st.spinner("Indexing policies…"):
                    _, count = rebuild_vector_store()
                st.success(f"Indexed {count:,} chunks")
            except Exception as exc:
                st.error(f"Failed: {exc}")


# ------------------------------------------------------------------------------
# App Header
# ------------------------------------------------------------------------------
doc_count = _document_count()
st.markdown(
    f"""
    <header class="app-header">
        <div class="header-left">
            <div class="header-logo">🛡️</div>
            <div>
                <h1 class="header-title">Sentinel Enterprise Policy Assistant</h1>
                <div class="header-subtitle">Verifiable policy guidance grounded in authorized organization documents</div>
            </div>
        </div>
        <div class="header-badges">
            <span class="badge-pill status-green">● Guardrails Active</span>
            <span class="badge-pill">⚡ Groq GPT-120</span>
            <span class="badge-pill">🔒 {doc_count} Approved Policies</span>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Main Content Tabs
# ------------------------------------------------------------------------------
chat_tab, policy_tab, eval_tab, red_tab = st.tabs(
    ["💬  Policy Assistant", "📑  Approved Documents", "📊  Quality Benchmark", "🎯  Security Probes"]
)


# --- TAB 1: Chat Assistant ---
with chat_tab:
    suggested_question = None

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-welcome">
                <h3>What would you like to know about company policy?</h3>
                <p>Select a common inquiry below or type your question. Every claim is validated against verified policy sources.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prompt_cols = st.columns(3)
        suggestions = [
            "What is the travel reimbursement policy?",
            "How should employees handle sensitive data?",
            "What are the rules for using AI tools at work?",
        ]
        for index, suggestion in enumerate(suggestions):
            if prompt_cols[index].button(f"💼 {suggestion}", key=f"suggestion_{index}", use_container_width=True):
                suggested_question = suggestion

    # Render message history
    for item in st.session_state.messages:
        if item["role"] == "user":
            _render_user_message(item["content"])
        else:
            with st.chat_message("assistant", avatar="🛡️"):
                st.markdown(item["content"])
                if "sources" in item and item["sources"]:
                    _render_sources(item["sources"])

    # Chat input
    typed_question = st.chat_input("Ask a question regarding HR, security, compliance, or travel policy…")
    question = typed_question or suggested_question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        _render_user_message(question)

        with st.chat_message("assistant", avatar="🛡️"):
            try:
                with st.spinner("Analyzing verified policies…"):
                    result = EnterprisePolicyAssistant(provider=selected_provider, model=selected_model).ask(question)

                st.markdown(result.answer)
                _render_trust_bar(result)
                _render_sources(result.sources)

                if result.block_reason:
                    st.warning(f"Guardrail Alert: {result.block_reason}")

                assistant_content = result.answer
                sources_to_save = result.sources
            except Exception as exc:
                assistant_content = f"Error: {exc}\n\nPlease ensure `GROQ_API_KEY` is configured in your `.env` file."
                sources_to_save = []
                st.error(assistant_content)

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_content,
            "sources": sources_to_save,
        })


# --- TAB 2: Approved Documents ---
with policy_tab:
    st.markdown("### Approved Corporate Policy Library")
    st.caption("These documents constitute the ground-truth knowledge base for Sentinel. Modifications require administrative compliance approval.")
    
    doc_cols = st.columns(2)
    policies = sorted(list(settings.policy_dir.glob("*.md")))
    for idx, ppath in enumerate(policies):
        col_idx = idx % 2
        with doc_cols[col_idx]:
            content = ppath.read_text(encoding="utf-8", errors="ignore")
            title = ppath.name.replace("_", " ").replace(".md", "").title()
            with st.expander(f"📄 {title} (`{ppath.name}`)", expanded=False):
                st.markdown(content[:600] + ("…" if len(content) > 600 else ""))
                st.caption(f"Full document size: {len(content):,} characters")


# --- TAB 3: Quality Benchmark Lab ---
with eval_tab:
    st.markdown("### Retrieval & Accuracy Evaluation")
    st.caption("Benchmark response precision against the curated golden reference dataset.")
    
    ctrl_col, info_col = st.columns([1, 2])
    with ctrl_col:
        limit = st.slider("Evaluation cases", 1, 10, 5)
        run_eval = st.button("Run Benchmark Evaluation", type="primary", use_container_width=True)
    with info_col:
        st.info("Scores answers on reference correctness, answer relevance, source retrieval, and factual grounding.")

    if run_eval:
        try:
            with st.spinner("Running benchmark cases…"):
                df = run_evaluation(limit=limit, provider=selected_provider, model=selected_model)
            
            m_cols = st.columns(4)
            m_cols[0].metric("Retrieval Hit", f"{_safe_mean(df, 'retrieval_hit'):.0%}")
            m_cols[1].metric("Groundedness", f"{_safe_mean(df, 'groundedness'):.0%}")
            m_cols[2].metric("Relevance", f"{_safe_mean(df, 'answer_relevance'):.0%}")
            m_cols[3].metric("Correctness", f"{_safe_mean(df, 'correctness'):.0%}")

            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "retrieval_hit": st.column_config.CheckboxColumn("Retrieved"),
                    "groundedness": st.column_config.ProgressColumn("Grounding", min_value=0, max_value=1),
                    "answer_relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1),
                    "correctness": st.column_config.ProgressColumn("Correctness", min_value=0, max_value=1),
                },
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")


# --- TAB 4: Security Red Team ---
with red_tab:
    st.markdown("### Guardrail Resilience Testing")
    st.caption("Probe adversarial prompt injection resilience and privacy safeguards.")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.caption("Tests injection prevention, PII protection, and out-of-scope boundaries.")
    with col_b:
        run_red = st.button("Launch Security Probes", type="primary", use_container_width=True)

    if run_red:
        try:
            with st.spinner("Running security probes…"):
                df = run_red_team(provider=selected_provider, model=selected_model)

            passed = int(df["passed"].astype(bool).sum())
            res_cols = st.columns(3)
            res_cols[0].metric("Pass Rate", f"{_safe_mean(df, 'passed'):.0%}")
            res_cols[1].metric("Defenses Passed", f"{passed} / {len(df)}")
            res_cols[2].metric("Flagged Probes", len(df) - passed)

            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "passed": st.column_config.CheckboxColumn("Passed"),
                    "blocked": st.column_config.CheckboxColumn("Blocked"),
                    "pii_redacted": st.column_config.CheckboxColumn("PII Sanitized"),
                },
            )
        except Exception as exc:
            st.error(f"Security probe failed: {exc}")

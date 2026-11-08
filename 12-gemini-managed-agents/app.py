"""Interactive Streamlit console for locally managed Gemini agent sessions."""

import os
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import errors

from agent import AgentDefinition, AgentSession, SessionStore, run_turn


DATABASE_PATH = Path(__file__).with_name("agent_sessions.db")
MODELS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
}


@st.cache_resource
def get_store() -> SessionStore:
    return SessionStore(DATABASE_PATH)


def session_label(session: AgentSession) -> str:
    return f"{session.name} · {session.model} · {session.created_at[:10]}"


def render_conversation(store: SessionStore, session_id: str) -> None:
    for message in store.history(session_id):
        role = "assistant" if message.role == "model" else "user"
        with st.chat_message(role):
            for part in message.parts or []:
                if part.text:
                    st.markdown(part.text)


def render_events(store: SessionStore, session_id: str) -> None:
    events = store.events(session_id)
    if not events:
        st.info("No events recorded for this session yet.")
        return

    for event in reversed(events):
        label = event["event_type"].replace(".", " · ").title()
        with st.container(border=True):
            st.caption(f'{event["created_at"]}  ·  {label}')
            if event["content"]:
                st.markdown(event["content"])


def main() -> None:
    st.set_page_config(
        page_title="Gemini Agent Studio",
        page_icon="✳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --accent: #dc7546;
            --ink: #202426;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 82% 0%, rgba(220,117,70,.10), transparent 36%),
                linear-gradient(180deg, #fbfaf7 0%, #f5f3ef 100%);
        }
        [data-testid="stSidebar"] {
            background: #f0eee9;
            border-right: 1px solid rgba(32,36,38,.12);
        }
        .hero {
            padding: 1.8rem 2rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(32,36,38,.14);
            border-left: 5px solid var(--accent);
            border-radius: 14px;
            background: rgba(255,255,255,.78);
        }
        .eyebrow {
            color: var(--accent);
            font-size: .78rem;
            font-weight: 750;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .hero h1 {
            color: var(--ink);
            margin: .25rem 0 .45rem;
            font-size: clamp(2rem, 4vw, 3rem);
        }
        .hero p {
            color: #555b5d;
            margin-bottom: 0;
            max-width: 760px;
        }
        [data-testid="stMetric"] {
            padding: .9rem 1rem;
            border: 1px solid rgba(32,36,38,.11);
            border-radius: 12px;
            background: rgba(255,255,255,.72);
        }
        </style>
        <section class="hero">
            <div class="eyebrow">Episode 12 · Gemini API</div>
            <h1>Gemini Agent Studio</h1>
            <p>Define an agent once. Reopen durable conversations, inspect every
            event, and keep the runtime under your control.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    store = get_store()
    with st.sidebar:
        st.markdown("### Agent workspace")
        st.caption("Create a reusable agent configuration and a durable session.")
        with st.form("create_session_form", clear_on_submit=True):
            agent_name = st.text_input("Agent name", value="gemini-assistant")
            model_label = st.selectbox("Gemini model", options=list(MODELS))
            instruction = st.text_area(
                "System instruction",
                value="You are a helpful assistant. Be accurate, clear, and concise.",
                height=130,
            )
            create_submitted = st.form_submit_button(
                "＋  Create agent session",
                type="primary",
                use_container_width=True,
            )

        if create_submitted:
            if not agent_name.strip() or not instruction.strip():
                st.error("Agent name and system instruction are required.")
            else:
                new_session = store.create_session(
                    AgentDefinition(
                        name=agent_name.strip(),
                        model=MODELS[model_label],
                        system_instruction=instruction.strip(),
                    )
                )
                st.session_state["session_selector"] = new_session.session_id
                st.rerun()

        sessions = store.list_sessions()
        if sessions:
            session_by_id = {session.session_id: session for session in sessions}
            selected_id = st.selectbox(
                "Saved sessions",
                options=list(session_by_id),
                format_func=lambda session_id: session_label(
                    session_by_id[session_id]
                ),
                key="session_selector",
            )
            selected_session = session_by_id[selected_id]
        else:
            selected_session = None
            st.info("Create your first agent session to get started.")

        st.divider()
        st.caption("Runtime")
        st.markdown("**Persistence:** local SQLite")
        st.markdown("**Model calls:** Gemini API")
        st.caption(f"Database: `{DATABASE_PATH.name}`")

    if selected_session is None:
        st.markdown("### Your agent runtime starts here")
        st.write(
            "Create a session in the sidebar. Its configuration and conversation "
            "history are saved locally, so they remain available after a restart."
        )
        st.info(
            "This is a self-hosted managed-style demo, not a Gemini-hosted "
            "managed-agent service."
        )
        return

    messages = store.history(selected_session.session_id)
    events = store.events(selected_session.session_id)
    message_count = sum(event["event_type"] == "message" for event in events)
    error_count = sum(event["event_type"] == "request.error" for event in events)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Agent", selected_session.name)
    metric_columns[1].metric("Model", selected_session.model)
    metric_columns[2].metric("Messages", message_count)
    metric_columns[3].metric("Recorded errors", error_count)

    st.caption(
        f"Session `{selected_session.session_id}` · created "
        f"{selected_session.created_at}"
    )
    conversation_tab, events_tab, details_tab = st.tabs(
        ["Conversation", "Event timeline", "Agent configuration"]
    )

    with conversation_tab:
        render_conversation(store, selected_session.session_id)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.warning(
                "Set `GEMINI_API_KEY` in the environment before sending a message."
            )
        prompt = st.chat_input(
            "Message this agent...",
            disabled=not bool(api_key),
        )
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                client = genai.Client(api_key=api_key)
                with st.chat_message("assistant"):
                    with st.spinner("Gemini is thinking..."):
                        answer = run_turn(
                            selected_session.session_id,
                            prompt,
                            store=store,
                            client=client,
                        )
                    st.markdown(answer)
                st.rerun()
            except errors.APIError as error:
                st.error(f"Gemini API request failed: {error}")
            except RuntimeError as error:
                st.error(str(error))

    with events_tab:
        st.markdown("### Session event stream")
        st.caption(
            "Events are persisted locally in order. User prompts and assistant "
            "responses are included."
        )
        render_events(store, selected_session.session_id)

    with details_tab:
        st.markdown("### Agent definition")
        st.markdown(f"**Name:** {selected_session.name}")
        st.markdown(f"**Model:** `{selected_session.model}`")
        st.markdown("**System instruction**")
        st.code(selected_session.system_instruction, language="text")
        st.info(
            "This demo runs the session loop in your Streamlit process and "
            "stores state in a local SQLite file. It does not provide a hosted "
            "sandbox, background scheduler, encrypted credential vault, or "
            "cloud crash recovery."
        )

    st.caption(
        "Privacy: prompts and replies are sent to Gemini and stored in the "
        "local, unencrypted SQLite database."
    )


if __name__ == "__main__":
    main()

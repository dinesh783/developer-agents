"""A Gemini-backed agent with durable local sessions and an event history."""

import argparse
import json
import os
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from google import genai
from google.genai import types


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    model: str
    system_instruction: str


@dataclass(frozen=True)
class AgentSession:
    session_id: str
    name: str
    model: str
    system_instruction: str
    created_at: str


class SessionStore:
    """Persist agent configuration and session events in a local SQLite file."""

    def __init__(self, database_path: Path = Path("agent_sessions.db")) -> None:
        self.database_path = Path(database_path)
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    system_instruction TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    event_type TEXT NOT NULL,
                    role TEXT,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_by_session
                    ON events(session_id, id);
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_session(self, agent: AgentDefinition) -> AgentSession:
        session_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        session = AgentSession(
            session_id=session_id,
            name=agent.name,
            model=agent.model,
            system_instruction=agent.system_instruction,
            created_at=created_at,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions
                    (id, agent_name, model, system_instruction, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.name,
                    session.model,
                    session.system_instruction,
                    session.created_at,
                ),
            )
            self._append_event(
                connection,
                session.session_id,
                "session.created",
                None,
                json.dumps({"agent": session.name, "model": session.model}),
            )
        return session

    def get_session(self, session_id: str) -> AgentSession:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Agent session not found: {session_id}")
        return AgentSession(
            session_id=row["id"],
            name=row["agent_name"],
            model=row["model"],
            system_instruction=row["system_instruction"],
            created_at=row["created_at"],
        )

    def list_sessions(self) -> list[AgentSession]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [
            AgentSession(
                session_id=row["id"],
                name=row["agent_name"],
                model=row["model"],
                system_instruction=row["system_instruction"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def history(self, session_id: str) -> list[types.Content]:
        self.get_session(session_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM events
                WHERE session_id = ? AND event_type = 'message'
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [
            types.Content(
                role=row["role"],
                parts=[types.Part.from_text(text=row["content"])],
            )
            for row in rows
        ]

    def record_message(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "model"}:
            raise ValueError("Message role must be 'user' or 'model'.")
        with self._connection() as connection:
            self._append_event(connection, session_id, "message", role, content)

    def record_error(self, session_id: str, error_type: str) -> None:
        with self._connection() as connection:
            self._append_event(
                connection, session_id, "request.error", None, error_type
            )

    def events(self, session_id: str) -> list[dict[str, Any]]:
        self.get_session(session_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, event_type, role, content, created_at
                FROM events WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        session_id: str,
        event_type: str,
        role: str | None,
        content: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events (session_id, event_type, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event_type,
                role,
                content,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def run_turn(
    session_id: str,
    prompt: str,
    *,
    store: SessionStore,
    client: Any,
) -> str:
    """Send one turn to Gemini and retain the user/assistant event history."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    session = store.get_session(session_id)
    store.record_message(session_id, "user", prompt)
    try:
        response = client.models.generate_content(
            model=session.model,
            contents=store.history(session_id),
            config=types.GenerateContentConfig(
                system_instruction=session.system_instruction
            ),
        )
        answer = getattr(response, "text", None)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Gemini returned no text response.")
        store.record_message(session_id, "model", answer)
        return answer
    except Exception as error:
        store.record_error(session_id, type(error).__name__)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Gemini-backed agents with durable local sessions."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("agent_sessions.db"),
        help="SQLite session database (default: agent_sessions.db)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="Create a durable agent session.")
    create.add_argument("--name", default="gemini-managed-agent")
    create.add_argument("--model", default="gemini-2.5-flash")
    create.add_argument(
        "--instruction",
        default="You are a helpful assistant. Be accurate, clear, and concise.",
    )

    chat = commands.add_parser("chat", help="Send a prompt to an existing session.")
    chat.add_argument("session_id")
    chat.add_argument("prompt", nargs="+")

    events = commands.add_parser("events", help="Show a session's event timeline.")
    events.add_argument("session_id")
    commands.add_parser("sessions", help="List saved sessions.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    store = SessionStore(args.database)

    if args.command == "create":
        session = store.create_session(
            AgentDefinition(args.name, args.model, args.instruction)
        )
        print(f"Created session: {session.session_id}")
        print(f"Agent: {session.name} ({session.model})")
        return 0

    if args.command == "sessions":
        for session in store.list_sessions():
            print(f"{session.session_id}  {session.name}  {session.model}")
        return 0

    if args.command == "events":
        for event in store.events(args.session_id):
            content = event["content"].replace("\n", " ")
            if len(content) > 120:
                content = content[:117] + "..."
            print(
                f'{event["id"]:>4}  {event["created_at"]}  '
                f'{event["event_type"]}  {content}'
            )
        return 0

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set the GEMINI_API_KEY environment variable.", file=sys.stderr)
        return 2
    answer = run_turn(
        args.session_id,
        " ".join(args.prompt),
        store=store,
        client=genai.Client(api_key=api_key),
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

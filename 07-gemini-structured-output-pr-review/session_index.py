import json
import os
import tempfile
from pathlib import Path
from uuid import UUID

INDEX_PATH = Path(__file__).parent / ".pr_sessions.json"
SESSION_DIR = Path(__file__).parent / ".sessions"


def _load_index() -> dict[str, str]:
    if not INDEX_PATH.exists():
        return {}
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read session index {INDEX_PATH}: {error}") from error
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in data.items()
    ):
        raise RuntimeError(f"Session index {INDEX_PATH} must map PR numbers to session IDs.")
    return data


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def get_session_id(pr_number: int) -> str | None:
    if isinstance(pr_number, bool) or pr_number < 1:
        raise ValueError("pr_number must be a positive integer.")
    return _load_index().get(str(pr_number))


def remember_session(pr_number: int, session_id: str) -> None:
    if isinstance(pr_number, bool) or pr_number < 1:
        raise ValueError("pr_number must be a positive integer.")
    try:
        normalized_id = str(UUID(session_id))
    except (ValueError, AttributeError) as error:
        raise ValueError("session_id must be a valid UUID.") from error
    index = _load_index()
    index[str(pr_number)] = normalized_id
    _write_json(INDEX_PATH, index)


def forget_session(pr_number: int) -> None:
    if isinstance(pr_number, bool) or pr_number < 1:
        raise ValueError("pr_number must be a positive integer.")
    index = _load_index()
    index.pop(str(pr_number), None)
    _write_json(INDEX_PATH, index)


def load_session(session_id: str) -> dict[str, object] | None:
    try:
        normalized_id = str(UUID(session_id))
    except (ValueError, AttributeError) as error:
        raise ValueError("session_id must be a valid UUID.") from error
    path = SESSION_DIR / f"{normalized_id}.json"
    if not path.exists():
        return None
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read saved session {path}: {error}") from error
    if not isinstance(session, dict) or session.get("session_id") != normalized_id:
        raise RuntimeError(f"Saved session {path} has an invalid format.")
    return session


def save_session(session: dict[str, object]) -> None:
    session_id = session.get("session_id")
    if not isinstance(session_id, str):
        raise ValueError("session must contain a UUID session_id.")
    try:
        normalized_id = str(UUID(session_id))
    except ValueError as error:
        raise ValueError("session_id must be a valid UUID.") from error
    if normalized_id != session_id:
        raise ValueError("session_id must use canonical UUID form.")
    _write_json(SESSION_DIR / f"{normalized_id}.json", session)

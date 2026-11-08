# Gemini Managed-Style Agents — Episode 12

This project turns the deck's managed-agent architecture into an interactive
Gemini API demo. The Streamlit console lets you create agent definitions,
continue saved conversations, and inspect the persisted event timeline. It
demonstrates the same four useful primitives:

- **Agent:** a reusable name, Gemini model, and system instruction.
- **Environment:** API credentials and the session database are configured
  outside the source code.
- **Session:** conversation history and the agent configuration survive
  process restarts in SQLite.
- **Events:** session creation, user/model messages, and request failures are
  stored in an ordered event timeline.

This is a small, **self-hosted managed-style runtime**, not a Gemini-hosted
managed-agent product and not Anthropic's Managed Agents service. The local
application owns persistence and lifecycle; Gemini provides model generation.
There is no built-in cloud sandbox, scheduling, crash recovery worker, or
encrypted credential vault. Deployments that need those guarantees must add
and operate the appropriate infrastructure.

## Requirements

- Python 3.10 or newer
- Streamlit 1.57 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Set up on Windows (PowerShell)

From this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-gemini-key"
```

Keep the API key in the environment; never put it in source code or commit it.
The default database is `agent_sessions.db` and is ignored by Git. It contains
conversation text and system instructions, so protect it like application
data.

## Run

Launch the interactive app:

```powershell
python -m streamlit run app.py
```

Create an agent session in the sidebar, select it, and chat. The conversation
and event timeline are saved in the local `agent_sessions.db` database.

The original command-line interface is also available:

Create a persisted agent session:

```powershell
python agent.py create --name support-agent --model gemini-2.5-flash
```

Copy the session ID from the output, then continue the same conversation—even
after restarting the program:

```powershell
python agent.py chat SESSION_ID "How do durable agent sessions work?"
python agent.py chat SESSION_ID "Summarize your previous answer."
```

List sessions or inspect an event timeline:

```powershell
python agent.py sessions
python agent.py events SESSION_ID
```

Use another database with the global option before the command:

```powershell
python agent.py --database .\demo.sqlite3 create --name demo-agent
```

## Privacy and limitations

The prompt is sent to the Gemini API. Prompts and responses are stored locally
in SQLite to enable durable session history; the event listing displays message
text. The database is not encrypted. Choose and protect its location
accordingly. API keys are read only from `GEMINI_API_KEY`.

The app records API failure event types and re-raises request errors. Failed
requests do not become successful-looking assistant messages. Sessions are
durable on the local machine, but this example does not claim cloud durability
or production availability.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

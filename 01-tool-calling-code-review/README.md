# Gemini Code Review Agent — Episode 01

A small Python tool-calling demo converted from the provided Claude API
snippets to Google's Gemini API. It asks Gemini to inspect files in a tiny
in-memory repository and returns a code-review-style answer.

## What you need

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

The Gemini API may have a no-cost tier, subject to current model availability,
quotas, and Google’s terms. Check the AI Studio usage and billing pages before
using it; a free key does not guarantee unlimited or permanently free access.

## Set up on Windows (PowerShell)

From this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key-here"
python agent.py
```

The SDK reads `GEMINI_API_KEY` from the environment. For a persistent local
setup, run `setx GEMINI_API_KEY "your-key-here"` and open a new terminal.
`.env.example` is a template only; this starter does not load `.env` files
automatically. Never commit or share your API key.

See Google's [function-calling guide](https://ai.google.dev/gemini-api/docs/function-calling)
and [model list](https://ai.google.dev/gemini-api/docs/models) for current
API and model details.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

## How it works

`agent.py` declares the `read_repo_file` tool to Gemini. The agent loop sends
the conversation, runs each requested tool locally, returns tool results to
Gemini, and stops when Gemini responds with text. The repository is currently
an in-memory dictionary by default; no project files are read or modified
unless you explicitly pass a repository folder with `--repo-root`. In that
mode, the agent can read UTF-8 text files inside that folder, up to 1 MB each.
It does not edit files. Avoid pointing it at folders containing secrets, since
file contents selected by Gemini are sent to the Gemini API for review.

For example, to review the project in the current directory:

```powershell
python agent.py --repo-root . "Summarize this project and identify its main risks."
```

To try other questions, pass a different question on the command line or
import and call `run_agent("your question", repo_root=Path(r"D:\path\to\repo"))`
from Python.

# Gemini Repo & PR Review Agent — Episode 05

Episode 05 adapts the supplied multi-agent coordinator example to the Gemini
function-calling pattern used in EP1–EP4. Gemini has a different agent API from
Claude's SDK, so `agent.py` implements two isolated specialist conversations
with narrowly scoped local tools and runs both specialist requests concurrently.
Each specialist gets its local tool result as data and uses one Gemini request
to report its findings, reducing request use under per-minute quotas.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Set up on Windows (PowerShell)

From this folder (`05-gemini-multi-agent-pr-review`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key-here"
python agent.py
```

The Google Gen AI SDK reads `GEMINI_API_KEY` from the environment. For local
persistence, use `setx GEMINI_API_KEY "your-key-here"` and open a new terminal.
Never put the key in source code or commit it.

## What the agent does

The coordinator asks two specialists to run in parallel:

- `docstring-reviewer` can only use `check_docstrings` to inspect a Python file.
- `security-reviewer` can only use `fetch_pr_metadata` to inspect mock PR data.

Each specialist starts with a new conversation and receives only its own task.
The coordinator combines their results and can post one mock review comment.
Comment evidence is validated before dispatch. The existing EP4 command and
PR-metadata guardrails are carried forward; no shell command is executed.
PR metadata and comments are mock data, not a real review-service integration.

The default prompt uses `01-tool-calling-code-review/agent.py` and mock PR
#143, whose body contains instruction-like text for the security reviewer to
detect and treat as untrusted data.

If Gemini returns HTTP 429 (`RESOURCE_EXHAUSTED`), the agent prints a concise
quota message rather than an SDK traceback. Wait for the retry interval shown
in the provider details, then rerun. The pip upgrade notice is informational
and unrelated to Gemini API quotas.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

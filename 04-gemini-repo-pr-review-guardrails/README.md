# Gemini Repo & PR Review Agent — Episode 04

Episode 04 adapts the supplied Hooks & Guardrails example to the Gemini
function-calling pattern used in EP1–EP3. Gemini does not use Claude SDK hook
callbacks, so `agent.py` applies equivalent checks in its local tool dispatcher.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Set up on Windows (PowerShell)

From this folder (`04-gemini-repo-pr-review-guardrails`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key-here"
python agent.py
```

The Google Gen AI SDK reads `GEMINI_API_KEY` from the environment. To persist
the variable locally, use `setx GEMINI_API_KEY "your-key-here"` and open a new
terminal. Never place the key in source code or commit it.

## What the agent demonstrates

- `check_docstrings` reads a UTF-8 Python file under the repository root,
  rejects paths outside it, enforces a 1 MB limit, and reports undocumented
  top-level functions and classes.
- `fetch_pr_metadata` returns mock PR data; the dispatcher normalizes its
  timestamp and state and labels the PR body as untrusted external content.
- `post_review_comment` rejects missing or invalid file/line/comment evidence.
  It returns a mock confirmation and does not post to a real service.
- `check_bash_command` blocks a few known-dangerous command patterns. It is a
  validation-only demonstration: it never executes shell commands, and its
  regular expressions are not a complete security boundary.

The default prompt reviews mock PR #143. Its embedded instruction-like text is
treated as untrusted data; because the mock contains no diff, the agent is
asked not to invent a code finding.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

# 07 - Gemini Structured-Output PR Review Agent

This project builds the Episode 07 structured-output review flow using the Google Gen AI SDK and `GEMINI_API_KEY`.

## Setup

```powershell
cd 07-gemini-structured-output-pr-review
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key-here"
```

## Run

```powershell
python agent.py 42
python agent.py 42 "Review this PR and submit structured findings."
```

The app persists local review sessions in `.pr_sessions.json` and `.sessions/` so a later run can resume the same PR conversation.

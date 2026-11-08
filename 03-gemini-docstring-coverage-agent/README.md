# Gemini Docstring Coverage Agent — Episode 03

Episode 03 adds a custom AST-based tool to the Gemini agent. It checks Python
files in this repository for functions and classes without docstrings, then
asks Gemini to assess the results for
`01-tool-calling-code-review/agent.py` and
`02-agent-code-summarizer/agent.py`.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

Gemini model availability, quotas, and pricing depend on Google's current
terms. Check AI Studio before use.

## Set up on Windows (PowerShell)

From this project folder (`03-gemini-docstring-coverage-agent`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key-here"
python agent.py
```

The Google Gen AI SDK reads `GEMINI_API_KEY` from the environment. To persist
the variable locally, run `setx GEMINI_API_KEY "your-key-here"` and open a new
terminal. Never commit or share your API key.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

The `check_docstring_coverage` tool uses Python's AST parser instead of text
search, skips private definitions by default, and reports missing docstrings
with line numbers. It reads only UTF-8 Python files inside the project root,
rejects paths that resolve outside it, and limits files to 1 MB. File contents
selected by Gemini are sent to the Gemini API for analysis; do not use this
agent with repositories containing secrets.

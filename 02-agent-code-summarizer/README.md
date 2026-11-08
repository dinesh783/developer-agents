# Gemini Agent — Episode 02

Rebuild the Episode 01 repository-review demo using Google's Gemini API. The
agent uses Gemini function calling to read
`01-tool-calling-code-review/agent.py` and return a concise, three-bullet
summary.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

Gemini API availability, quotas, and pricing depend on the selected model and
Google's current terms. Check AI Studio before use.

## Set up on Windows (PowerShell)

From this project folder (`02-agent-code-summarizer`):

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

See Google's [function-calling guide](https://ai.google.dev/gemini-api/docs/function-calling)
and [model list](https://ai.google.dev/gemini-api/docs/models) for API details.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

The agent reads only the sibling
`01-tool-calling-code-review/agent.py` file and does not edit project files.
The source contents are sent to Gemini for summarization.

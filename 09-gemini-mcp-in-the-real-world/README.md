# Gemini MCP in the Real World — Episode 09

This episode adapts the MCP companion example to the Gemini API used by the
earlier projects. The agent bridges Gemini function calls to MCP tools:

- `repo_facts` is a separate, local MCP server. Its `count_todos` tool scans
  Python files under this project and reports TODO/FIXME comments.
- `github` is an optional remote GitHub MCP server. It is connected only for
  `--live` questions, and the agent exposes only its read-only
  `pull_request_read` tool to Gemini.

The server definitions live in `.mcp.json`. Unlike clients that discover this
file themselves, this Gemini example reads it and bridges MCP tool schemas and
tool results to Gemini's function-calling API.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- A GitHub personal access token with read-only access to pull requests for
  live GitHub questions

## Set up on Windows (PowerShell)

From this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-gemini-key"
```

To use live GitHub data, also set the token:

```powershell
$env:GITHUB_PAT = "your-read-only-token"
```

Never put either key in `.mcp.json`, source code, or a committed file.

## Run

Scan this workspace with the local MCP tool:

```powershell
python agent.py
python agent.py "Count TODOs under tests."
```

Ask a question using the optional remote GitHub MCP server:

```powershell
python agent.py --live "What are the 3 most recently updated open pull requests on anthropics/claude-agent-sdk-python?"
```

The live path requires both `GEMINI_API_KEY` and `GITHUB_PAT`, a working
network connection, and access to the configured GitHub MCP endpoint. Server
or API errors are reported instead of being treated as successful responses.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

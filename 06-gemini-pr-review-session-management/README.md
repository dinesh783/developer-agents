# Gemini Repo & PR Review Agent — Episode 06

Episode 06 adapts the supplied memory, sessions, and context-management example
to the Gemini function-calling agent built in EP5. The source snippet uses the
Claude Agent SDK, whose session and hook APIs are not available in Gemini. This
version implements the same ideas explicitly with local JSON session storage,
Gemini function calling, and a local compaction log.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Set up on Windows (PowerShell)

From this folder (`EP6`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key-here"
python agent.py 42
```

The Google Gen AI SDK reads `GEMINI_API_KEY` from the environment. To persist
it locally, use `setx GEMINI_API_KEY "your-key-here"` and open a new terminal.
Never put the key in source code or commit it.

## What this episode adds

- `GEMINI.md` holds project instructions which `agent.py` explicitly loads
  into the Gemini system instruction.
- `.pr_sessions.json` maps a PR number to a local session UUID. Full session
  context is stored under `.sessions/`; both are ignored by Git.
- A later run for the same PR restores saved context and appends the new turn.
  Use `--fresh` to start over, or `--fork` to create a separate branch of the
  saved context without changing the PR's normal session pointer.
- When a saved session grows past four turns, older turns are summarized and
  recorded in `.compactions.log`. This is application-managed context
  compaction, not a Gemini SDK hook.
- `context_check.py` uses Gemini's `count_tokens` API to show approximate token
  counts for the system instructions and sample context categories.

The mock PR data and review comments are local demonstrations, not a real
GitHub integration. `check_bash_command` only validates text and never runs it.
Gemini's server-side conversation state is not used; all resumable context is
stored locally by this app.

## Demo

```powershell
python agent.py 42
python agent.py 42 "Summarize what the earlier review found."
python agent.py 42 --fork "Try an alternative review approach."
python agent.py 42 --fresh
python context_check.py
```

Remove `.pr_sessions.json`, `.sessions`, and `.compactions.log` to reset local
demo state. These files contain review prompts and responses; keep them private.

## Tests

```powershell
python -m unittest discover -s tests -t .
```

## AI evaluations

The unit tests use mocked Gemini responses and verify deterministic application
behavior. To check the real security-review specialist against two repeatable
PR-description scenarios, set `GEMINI_API_KEY` and run:

```powershell
python -m evals.run_evals
```

The scenarios check that PR #42's embedded instruction is identified and
quoted, while benign PR #142 is not falsely reported as an injection. The
runner makes two Gemini calls and saves a JSON report to
`evals/results/latest.json` by default; override the location with `--output`.
Evaluation calls use API quota and model responses can vary. This initial
keyword-based rubric is a baseline, not a comprehensive quality or security
assessment; inspect failed and borderline reports before changing prompts.

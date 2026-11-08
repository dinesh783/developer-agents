# Gemini Model Selection — Episode 10

This episode turns the model-selection lesson into a runnable Gemini API demo.
It chooses a model tier per request by task type and stakes, then sends the
prompt to Gemini. The three tiers illustrate the deck's capability, cost, and
latency tradeoff:

| Tier | Default model | Example use |
| --- | --- | --- |
| Fast | `gemini-2.5-flash-lite` | High-volume classification, extraction, and formatting |
| Balanced | `gemini-2.5-flash` | Code review and everyday multi-purpose work |
| Capable | `gemini-2.5-pro` | Multi-step reasoning and high-stakes tasks |

Model IDs are defaults, not promises that an ID is available for every API
key or region. Check the current [Gemini model list](https://ai.google.dev/gemini-api/docs/models)
and override IDs with environment variables when needed. The effort option
sets response-depth instructions and can raise the chosen tier; it does not
send a model-specific thinking-budget parameter. The API's own reasoning
behavior can differ by model.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Set up on Windows (PowerShell)

From this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-gemini-key"
```

Never put the API key in source code or a committed file.

## Run

```powershell
python agent.py --task classify --stakes low "Classify this message as urgent or not: ..."
python agent.py --task code-review --effort medium "Review this function for bugs: ..."
python agent.py --task multi-step --stakes high "Compare these architecture choices: ..."
```

Supported tasks are `classify`, `extract`, `format`, `code-review`,
`multi-step`, and `open-ended`. Stakes are `low`, `medium`, or `high`. Effort
is `auto`, `low`, `medium`, or `high`. High stakes and explicit effort may
raise the recommended tier but never lower the task's baseline tier.

Configure or replace the tier model IDs without editing code:

```powershell
$env:GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"
$env:GEMINI_MODEL_BALANCED = "gemini-2.5-flash"
$env:GEMINI_MODEL_CAPABLE = "gemini-2.5-pro"
```

Use `--model MODEL_ID` to override the recommendation for an individual run.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

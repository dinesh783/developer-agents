# Gemini Production Operations — Episode 11

This companion project turns Episode 11's operational themes into a small,
runnable Gemini API example:

- Captures per-request latency and Gemini-reported input, output, and cached
  token counts.
- Estimates token charges using rates you supply, rather than embedding
  potentially stale prices in the code.
- Optionally creates a Gemini context cache for stable system instructions.
- Appends prompt-free JSONL events for success and API failures.
- Surfaces API errors instead of treating them as successful responses.

The supplied deck includes Anthropic-specific model names, SDK billing notes,
and cache figures. This project uses Gemini's API and its usage metadata
instead; pricing and caching behavior vary by model and current Google terms.
The estimate covers token charges for a generation only. Check Google's
[Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) for current
rates and any cache storage charges before configuring rates.

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

Never put the API key in source code or a committed file. The program reads it
from `GEMINI_API_KEY`.

To estimate token cost, set rates in USD per million tokens using the current
rate for the selected model:

```powershell
$env:GEMINI_INPUT_PRICE_PER_MILLION = "0.30"
$env:GEMINI_OUTPUT_PRICE_PER_MILLION = "2.50"
$env:GEMINI_CACHED_INPUT_PRICE_PER_MILLION = "0.03"
```

These example values are illustrative only. If rates are unset, the program
still reports token usage and latency, but explicitly marks cost as
unconfigured. If a response contains cached tokens, the cached-input rate is
also required to produce an estimate.

## Run

```powershell
python agent.py "Summarize the production trinity: cost, observability, debugging."
```

Use a stable context file to create an ephemeral Gemini context cache:

```powershell
python agent.py --cache-context-file .\system-instructions.txt --cache-ttl 3600s "Answer this request."
```

The cached context is sent to Gemini and the cache remains available until its
TTL expires. Use the current API documentation to check model support,
minimum-token requirements, and cache pricing. Requests and context may
contain sensitive information: the JSONL log intentionally omits prompt and
context text, but Gemini receives them to generate the response. The default
log file is `runs.jsonl`; choose another location with `--log-file`.

## Telemetry format

Each JSONL line contains a timestamp, model, status, latency, token usage when
available, and estimated per-generation token cost when rates are configured.
API failure events record the exception type, not the prompt or provider
message.

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

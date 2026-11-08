"""One-off: inspect a Gemini prompt's token usage by local context category."""

from agent import MODEL, SYSTEM_INSTRUCTION, _create_client


def main() -> None:
    """Count the system prompt and representative session input with Gemini."""
    client = _create_client()
    categories = {
        "system instructions": SYSTEM_INSTRUCTION,
        "review tools and task": (
            "Review PR #42. Delegate a docstring review and security review, "
            "then post one evidence-based comment."
        ),
        "saved-session context": (
            "Previous review: PR #42 body contained a prompt-injection attempt. "
            "The security reviewer identified it as untrusted content."
        ),
    }
    counts = {}
    for name, text in categories.items():
        response = client.models.count_tokens(model=MODEL, contents=text)
        counts[name] = response.total_tokens
    total = sum(counts.values())
    print(f"Approximate prompt context: {total:,} tokens")
    for name, tokens in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {name:<28}{tokens:>9,}")
    print("Counts are estimates for this sample; tool schemas and responses add tokens.")


if __name__ == "__main__":
    main()

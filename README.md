# answerbot

The simplified research agent: **one AI, one search service, one log file.**

Every question is always searched (never answered from memory), sources are
read in full, citations can only point to pages that were actually fetched,
and every step lands in `journal.jsonl`.

## Setup

1. Install Rust: https://rustup.rs
2. Get two API keys:
   - Anthropic: https://console.anthropic.com
   - Firecrawl: https://firecrawl.dev
3. Run:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export FIRECRAWL_API_KEY=fc-...
cargo run --release -- "what changed in the latest Rust edition?"
```

## How the code maps to the flow

| Flow step | Where in `src/main.rs` |
|---|---|
| You ask | `main()` reads the command line |
| AI writes one search query | first `llm()` call |
| Firecrawl finds + reads pages in one trip | `search()` |
| One re-search allowed if needed | the `SEARCH:` branch |
| Citations kept honest | the regex strip near the end |
| Everything noted in one file | `journal()` → `journal.jsonl` |

## Knobs (top of `main.rs`)

- `MAX_SOURCES_PER_SEARCH` — pages read per trip (default 4)
- `MAX_CHARS_PER_SOURCE` — truncation limit per page (default 8,000)
- `REQUEST_TIMEOUT_SECS` — hard cap per network call (default 30)

## Upgrade paths (when you need them, not before)

- **Streaming output**: switch the final `llm()` call to the Messages API
  streaming mode and print tokens as they arrive.
- **More depth per question**: raise the re-search cap from 1 to 2.

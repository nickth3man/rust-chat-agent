# answerbot

The simplified research agent: **one AI, one search service, one log file.**

Every question is always searched (never answered from memory), sources are
read in full, citations can only point to pages that were actually fetched,
and every step lands in `journal.jsonl`.

## Setup

1. Install Rust: https://rustup.rs
2. Get two API keys:
   - OpenRouter: https://openrouter.ai
   - Firecrawl: https://firecrawl.dev
3. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_MODEL` (e.g. `anthropic/claude-sonnet-4.5`)
   - `FIRECRAWL_API_KEY`
4. Run:

```bash
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

## Development

One-time tool install:

```bash
cargo install --locked just cargo-deny
```

Common tasks (see `justfile`):

```bash
just fmt        # format
just lint       # clippy, warnings are errors
just check      # type-check against Cargo.lock
just deny       # dependency advisories + license audit
just ci         # everything CI runs
just run "..."  # ask a question
```

Lints live in `Cargo.toml` under `[lints]` (clippy `all` + `pedantic` as
warnings); CI denies warnings, so keep `just lint` clean. Formatting is
rustfmt defaults plus `rustfmt.toml`. CI (`.github/workflows/ci.yml`) runs
fmt-check, clippy, check, and cargo-deny on every push and PR.

## Upgrade paths (when you need them, not before)

- **Streaming output**: switch the final `llm()` call to the Messages API
  streaming mode and print tokens as they arrive.
- **More depth per question**: raise the re-search cap from 1 to 2.

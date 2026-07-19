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
    - `FIRECRAWL_API_KEY`
4. Choose a model in `config/models.json` (model name and temperature).
5. Run:

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

## Configuration

Model selection and tuning live in `config/models.json`:

```json
{
  "model": "openai/gpt-oss-20b",
  "temperature": 0.7,
  "reasoning": true
}
```

- `model` — OpenRouter model id (browse all at https://openrouter.ai/models)
- `temperature` — sampling temperature; reasoning/thinking models ignore it
- `reasoning` — `true` sends `reasoning: {}` and journals the chain-of-thought;
  `false` (or omitted, defaulting to `true`) skips both. **Required for
  non-reasoning models** — some providers (e.g. Mistral via Cloudflare) reject
  the unknown parameter with HTTP 400.

Secrets (API keys) stay in `.env`.

## Model compatibility

Tested against four OpenRouter models at under ~$0.20/1M tokens:

| Model                                    | Reasoning | Citations per answer | Notes                       |
| ---------------------------------------- | --------- | -------------------- | --------------------------- |
| `openai/gpt-oss-20b`                       | yes       | ~1.9 avg             | requires `reasoning: true`    |
| `google/gemini-2.5-flash-lite`             | optional  | ~5.5 avg             | accepts `reasoning` but never returns it |
| `mistralai/mistral-small-3.2-24b-instruct` | no        | ~5.1 avg             | requires `reasoning: false`   |
| `meta-llama/llama-4-scout`                 | no        | ~4.3 avg             | requires `reasoning: false`   |

The answer loop has one zero-citation retry built in (`src/main.rs:368-388`):
if the model's first answer has no `[Sn]` citations despite having sources
available, it is re-prompted with a citation reminder. This catches
instruction-following gaps in smaller reasoning models. The retry is bounded —
exactly one, same as the `SEARCH:` re-search limit — to keep the structural
constraints from AGENTS.md intact.

Known flaky case: `openai/gpt-oss-20b` occasionally refuses the forced tool
call for the query-rewrite step on historical questions (e.g. "When did World
War II end?"), failing with `rewrite: model did not return a tool call`. This
is logged in `journal.jsonl` as a failure exit and is not retried; switch
models or rephrase the question if you hit it.

## Knobs (top of `main.rs`)

- `MAX_SOURCES_PER_SEARCH` — pages read per trip (default 4)
- `MAX_CHARS_PER_SOURCE` — truncation limit per page (default 8,000)
- `REQUEST_TIMEOUT_SECS` — hard cap per network call (default 30)

`REQUEST_TIMEOUT_SECS` bounds a single HTTP call, not the whole run. The
retry path can chain up to four sequential network calls (rewrite → search →
answer → re-search → answer → zero-citation retry), so a single CLI
invocation can take ~2 minutes in the worst case before any per-call timeout
fires. Anyone wrapping the CLI in an outer timeout (e.g. `tests/run_eval.py`
uses 60s) should account for this — slow models on the retry path will hit
the outer limit, not the per-call one.

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

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
4. Copy `config/models.json.example` to `config/models.json` and choose a
   model (model name and temperature). The real file is gitignored.
5. Run:

```bash
cargo run --release -- "what changed in the latest Rust edition?"
```

## How the code maps to the flow

| Flow step | Where in `src/main.rs` |
|---|---|
| You ask | `main()` reads the command line |
| AI writes one search query | `rewrite_llm()` (forced `generate_search_query` tool) |
| Firecrawl finds + reads pages in one trip | `search()` |
| AI answers from sources | `llm()` (plain chat — no tools) |
| One re-search allowed if needed | answer text starts with `SEARCH:` → host runs `search()` again |
| Citations kept honest | the regex strip near the end |
| Everything noted in one file | `journal()` → `journal.jsonl` (or `ANSWERBOT_JOURNAL`) |

There are two different “search” mechanisms:

- **Rewrite step** — the only OpenRouter tool call: `generate_search_query`,
  forced via `tool_choice: "required"`. It returns a short query string; it
  does not fetch the web.
- **Answer step** — no tools. If sources are insufficient, the model replies
  with exactly `SEARCH: <query>`; the host parses that line and calls
  Firecrawl once more (then re-asks with `insist=true`).

## Configuration

Model selection and tuning live in `config/models.json` (gitignored local
file — copy from `config/models.json.example`; there is no built-in default):

```json
{
  "model": "your-openrouter-model-id",
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

The compatibility table below is a tested set, not a recommended default.

Secrets (API keys) stay in `.env`. Optional path overrides (CWD-relative unless
absolute):

- `ANSWERBOT_CONFIG` — models JSON (default `config/models.json`)
- `ANSWERBOT_JOURNAL` — journal file (default `journal.jsonl`)

Run from the repo root, or set these when the CWD differs.

## Model compatibility

Tested against four OpenRouter models at under ~$0.20/1M tokens:

| Model                                    | Reasoning | Citations per answer | Notes                       |
| ---------------------------------------- | --------- | -------------------- | --------------------------- |
| `openai/gpt-oss-20b`                       | yes       | ~1.9 avg             | requires `reasoning: true`    |
| `google/gemini-2.5-flash-lite`             | optional  | ~5.5 avg             | accepts `reasoning` but never returns it |
| `mistralai/mistral-small-3.2-24b-instruct` | no        | ~5.1 avg             | requires `reasoning: false`   |
| `meta-llama/llama-4-scout`                 | no        | ~4.3 avg             | requires `reasoning: false`   |

The answer loop has one zero-citation retry built in near the end of `main`
in `src/main.rs`: if the model's first answer has no `[Sn]` citations despite
having sources available, it is re-prompted with a citation reminder. This
catches instruction-following gaps in smaller reasoning models. The retry is
bounded — exactly one, same as the `SEARCH:` re-search limit — to keep the
structural constraints from AGENTS.md intact. A late `SEARCH:` after the one
allowed re-search is rejected *before* this citation retry (so it does not
waste an LLM call), and again after the retry if that path also returns
`SEARCH:`.

Three bounded retry loops catch weaker-model failure modes:

- `rewrite_llm` retries up to `REWRITE_MAX_ATTEMPTS` times when the rewrite
  step emits no tool call, malformed arguments, a missing `query` field, or
  a blank `query`. `openai/gpt-oss-20b` occasionally refuses the forced tool
  call on historical questions; the retry absorbs that without operator action.
- `llm` retries up to `LLM_MAX_ATTEMPTS` times when the answer comes back
  with empty `content` — common for reasoning models that finish their
  chain-of-thought but emit no final answer.
- `post_json` retries up to `NETWORK_MAX_ATTEMPTS` times on transient HTTP
  failures (timeouts, connection errors, HTTP 429, any 5xx). Non-retryable
  errors (other 4xx, body-decode failures) propagate immediately.

Every retry emits a journal event with the attempt number and a short reason
(`rewrite_retry`, `empty_answer_retry`, `network_retry`). All three use
exponential backoff (`backoff_ms` in `src/lib.rs`: 250, 500, 1000, 2000,
4000 ms, capped at 4 s) so a stuck upstream does not get hammered.

## Knobs (top of `main.rs`)

- `MAX_SOURCES_PER_SEARCH` — pages read per trip (default 4)
- `MAX_CHARS_PER_SOURCE` — truncation limit per page (default 8,000)
- `REQUEST_TIMEOUT_SECS` — hard cap per network call (default 30)
- `REWRITE_MAX_ATTEMPTS` — total attempts at the rewrite tool call (default 5)
- `LLM_MAX_ATTEMPTS` — total attempts at the answering call (default 3)
- `NETWORK_MAX_ATTEMPTS` — total attempts at each HTTP POST (default 3)

`journal.jsonl` (override with `ANSWERBOT_JOURNAL`) appends full questions,
queries, reasoning, and answers — treat it as sensitive local data.

`REQUEST_TIMEOUT_SECS` bounds a single HTTP call, not the whole run. With
retries enabled, each HTTP attempt can take up to three timeouts plus the
backoff schedule (default 30 s + 250 ms + 30 s + 500 ms + 30 s ≈ 91 s), and
each LLM call (`rewrite_llm`, `llm`) layers its own attempt cap on top
(`REWRITE_MAX_ATTEMPTS` / `LLM_MAX_ATTEMPTS`). The worst-case chain —
rewrite → search → answer → optional re-search → optional zero-citation
retry, with every retry exhausted — is on the order of **20–25 minutes**;
typical runs are well under a minute. Anyone wrapping the CLI in an outer
timeout (e.g. `tests/run_eval.py` uses 60 s) should account for this: slow
models on the retry path will hit the outer limit, not the per-call one.

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

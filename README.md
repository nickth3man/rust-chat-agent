# OpenRouter Research Agent — Rust

A terminal research agent built on Rig 0.40 and OpenRouter. It streams answers, can fan one query out to every enabled search backend, ranks deduplicated results with a separate model, fetches bounded page content, compacts conversation memory, and writes a structured session journal with an evidence ledger.

## Prerequisites

- Rust **1.88 or newer** (MSRV; Edition 2024).
- `rustup` with the stable toolchain. `rust-toolchain.toml` selects `stable` and installs `rustfmt` and `clippy`.
- An OpenRouter API key. Search-provider credentials are optional unless you explicitly enable a provider that requires one.
- Network access to OpenRouter and the enabled search providers. The default providers require neither Docker nor provider API keys.

## Setup

Copy both example files from the repository root:

```bash
cp config.example.toml config.toml
cp .env.example .env
```

PowerShell equivalent:

```powershell
Copy-Item config.example.toml config.toml
Copy-Item .env.example .env
```

Set the required key in `.env`:

```dotenv
OPENROUTER_API_KEY=your-openrouter-key
```

Keep secrets in `.env`, not `config.toml`. Both local files are ignored by Git.

Provider `user_agent` values use the public project URL as the contact identity.

## Models

Chat, ranking, and memory summarization are independent OpenRouter model lanes:

| Purpose | `config.toml` |
|---|---|
| Stream the conversation and call tools | `models.chat_id` |
| Select and order deduplicated search evidence | `models.rank_id` |
| Summarize evicted conversation memory | `models.summarize_id` |

The example config uses `openrouter/auto` for all three; each field can instead hold a different OpenRouter model ID:

```toml
[models]
chat_id = "openrouter/auto"
rank_id = "openrouter/auto"
summarize_id = "openrouter/auto"
chat_context_tokens = 12000
```

`chat_context_tokens` drives the active conversation-memory window. Ranking and summarization use bounded request/output policies owned by their runtime lanes.

Model IDs are loaded only from `config.toml`. Process environment variables take precedence over `.env` for secrets and provider endpoints.

## Search providers

`meta_search` queries **all providers enabled in `config.toml` concurrently**. Provider selection is startup configuration; the tool accepts only a query and does not expose per-call category or backend filters.

The example enables 23 backends that work without credentials. GitHub and Stack Exchange can optionally use tokens for higher provider limits. Four additional backends are disabled until their required key or endpoint is configured.

| Backend | Focus | Example state | Credential or endpoint |
|---|---|---:|---|
| DuckDuckGo | Broad web, HTML/best effort | Enabled | None |
| Stract | Broad web | Enabled | None |
| Marginalia | Small/indie web | Enabled | None |
| Mwmbl | Small/indie web | Enabled | None |
| Wiby | Small/classic web, best effort | Enabled | None |
| SearchMySite | Small/indie web | Enabled | None |
| Wikipedia | Knowledge | Enabled | None |
| Wikidata | Knowledge/entities | Enabled | None |
| OpenLibrary | Books | Enabled | None |
| Free Dictionary | Definitions | Enabled | None |
| arXiv | Academic preprints | Enabled | None |
| Crossref | DOI metadata | Enabled | None |
| Semantic Scholar | Academic literature | Enabled | None |
| PubMed | Biomedical literature | Enabled | None |
| Hacker News | Developer/news discussion | Enabled | None |
| GitHub | Repositories | Enabled | Optional `GITHUB_TOKEN` |
| Stack Exchange | Technical Q&A | Enabled | Optional `STACKEXCHANGE_KEY` |
| npm | JavaScript packages | Enabled | None |
| crates.io | Rust crates | Enabled | None |
| MDN | Web documentation, best effort | Enabled | None |
| GDELT | News | Enabled | None |
| Reddit | Discussion, best effort | Enabled | None |
| Lobsters | Technical discussion, best effort | Enabled | None |
| Brave | Broad web | Disabled; keyed opt-in | `BRAVE_API_KEY` |
| Mojeek | Broad web | Disabled; keyed opt-in | `MOJEEK_API_KEY` |
| SearXNG | Metasearch endpoint | Disabled; endpoint opt-in | `SEARXNG_BASE_URL` |
| Firecrawl | Web search | Disabled; keyed opt-in | `FIRECRAWL_API_KEY`; endpoint defaults to `https://api.firecrawl.dev/v2/search` |

To opt in, set the provider's `enable = true` in `config.toml` and set its required value in `.env`. Startup fails with the provider and variable name if an enabled keyed provider has no key or enabled SearXNG has no endpoint. Leaving optional GitHub or Stack Exchange credentials blank keeps those providers enabled in key-free mode.

Provider controls live under each `[providers.<name>]` table: `enable`, concurrency, minimum request interval, timeout, user agent, and—where applicable—the credential or base-URL environment variable.

## Run

From the repository root:

```bash
cargo run
```

For an optimized build:

```bash
cargo run --release
```

The runtime always reads `config.toml` and `.env` from the crate root. It prints answer text as it arrives, along with tool, provider, ranking, and memory-compaction activity. A transient OpenRouter stream failure is retried once only when no output has yet become visible.

## REPL commands

| Command | Behavior |
|---|---|
| `/help` | Show command help. |
| `/status` | Show public model, provider, search, fetch, session, and config-path status plus the current session path. Credentials are not displayed. |
| `/compact` | Force conversation-memory compaction, even below the automatic pressure threshold. |
| `/clear` | Clear and forget the in-memory conversation. The session journal remains on disk and records the clear event. |
| `/quit` | Exit. `quit` and `exit` are aliases. |

An empty line is ignored. Each prompt can use at most six Rig tool/completion turns.

## How a research turn works

1. **Fan-out:** `meta_search` sends the query to every enabled backend concurrently. The example allows up to five hits per backend, applies provider concurrency/rate limits, and uses a 20-second outer stage budget. Most default provider HTTP timeouts are six seconds; Firecrawl's is 15 seconds.
2. **Degrade independently:** a failed provider becomes a warning while successful providers continue. The tool fails when every provider fails, when no candidates remain, when ranking fails, or when the ranker selects nothing.
3. **Deduplicate:** URLs are normalized across scheme, `www.`, default ports, trailing slashes, common tracking parameters, and query ordering. Duplicate hits merge provider labels, source subtypes, and the more informative title/snippet; cross-engine agreement is retained as metadata.
4. **Rank:** the separate rank model receives fixed candidate IDs and must return every candidate exactly once with a selection, concise decision, and optional score. Its output is validated; ranking errors do not silently fall back to provider order. The example rank timeout is 20 seconds.
5. **Bound model input:** selected hits and ranking decisions are serialized under `search.model_output_bytes` (6,144 bytes by default). Lower-ranked trailing selections are removed first when needed.
6. **Fetch when needed:** `fetch_page` accepts one URL and an optional character cap. The example permits HTTP(S) HTML, plain text, XHTML, JSON, and XML; follows at most five redirects; times out after 20 seconds; rejects bodies over 2,000,000 bytes; converts HTML to compact text/Markdown; and caps output at 50,000 Unicode characters. A transient request or HTTP 408/429/5xx response receives at most one retry.
7. **Answer and sources:** the chat model answers from tool evidence. After a successful turn, selected search evidence is printed as a numbered source list, deduplicated by normalized URL; the model is instructed not to manufacture that list.

### Conversation memory

Memory is process-local and keyed by the session conversation ID. It uses a heuristic token window at 82% of `chat_context_tokens`, choosing Anthropic- or Gemini-oriented counting heuristics when the chat model ID indicates those families and the default heuristic otherwise.

Before web-tool turns enter memory, raw `meta_search` and `fetch_page` payloads are replaced with compact evidence records. Search records retain bounded title, URL, snippet, provider, and rank decision fields. Fetch records retain the URL and a short leading excerpt; non-web tool results remain lossless.

When the window is under pressure—or `/compact` is used—the summarize model turns evicted messages into a system summary. It has a 30-second timeout, retries once, and then inserts an explicit safe fallback stating how many messages were omitted. `/clear` removes the in-memory conversation; it does not erase the session file.

## Session JSON and evidence ledger

Every run creates `sessions/session-<process-id>-<unique-id>.json` by default. `/status` prints the exact path. The document format is `openrouter-chat-session`, version 1, with:

- `metadata`: session ID plus creation and update timestamps;
- `transcript`: user, assistant, and tool text;
- `events`: provider activity and results, ranking decisions, tool/completion activity, stream failures, and compaction state;
- `provenance`: a per-turn evidence ledger containing only rank-selected search sources, including normalized URL, title, original URL, supporting snippet, rank decision, provider labels, and source subtypes.

The journal is append-only in memory and is pretty-printed to an atomically replaced file after each logical mutation. Search activity includes normalized hit details and structured ranking decisions. The terminal source list is rendered from the selected provenance ledger, not from text invented by the assistant.

## Security and privacy

- Put credentials in `.env`. The OpenRouter key is mandatory; provider keys are resolved only for enabled providers.
- Session redaction settings must remain `true`; startup rejects a configuration that disables credential or authorization-header redaction.
- Before each session write, configured OpenRouter/provider secrets and credential-shaped assignments such as `Authorization`, `X-Subscription-Token`, `api-key`, `api_key`, and `token` are replaced with `[REDACTED]`. Terminal tool arguments are separately bounded and sanitized before display; stream errors are bounded and sanitized before event logging. Redaction is defense in depth, not a reason to publish session files: journals still contain prompts, answers, URLs, snippets, and fetched tool output.
- The agent preamble treats web and tool material as untrusted data and forbids following embedded instructions. `fetch_page` additionally wraps returned content in `<web_content url="…">…</web_content>` fencing, and the summarizer is told to ignore instructions in conversation history.
- Search queries are sent to every enabled provider. Prompts and tool context are sent through OpenRouter. Review the enabled-provider list and local session retention for your privacy requirements.
- `fetch_page` restricts schemes, media types, redirects, time, bytes, and characters, but it **intentionally allows loopback and private/local HTTP(S) addresses**. This is not SSRF protection.

## Verification

Run the offline project checks from the repository root:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo doc --no-deps
```

On Windows, the verified toolchain target is stable MSVC. If your default host is GNU or lacks usable linker tooling, add `+stable-x86_64-pc-windows-msvc` after `cargo`, for example:

```powershell
cargo +stable-x86_64-pc-windows-msvc test --all-targets
```

The Rig/OpenRouter live stream contract is ignored by default. To run it intentionally, export `OPENROUTER_API_KEY` into the process environment and run:

```bash
cargo test --test p1_rig_contract p1_live_stream_and_memory_identity -- --ignored --exact
```

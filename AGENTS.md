# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

`answerbot` — a single-binary Rust CLI that answers a question by (1) asking an
LLM (via OpenRouter) to rewrite it into a search query, (2) searching + reading
pages with Firecrawl, and (3) answering with `[S1]`-style citations that are
validated against the actually-fetched sources. The orchestration (env loading,
LLM/Firecrawl calls, journaling, printing) lives in [src/main.rs](src/main.rs);
the pure LLM-facing helpers (`Source`, prompt formatting, citation validation)
live in [src/lib.rs](src/lib.rs). [README.md](README.md) maps each flow step to
the code.

Deliberate design constraints (do not "fix" these without being asked):

- **One re-search.** The answer loop allows exactly one `SEARCH:` retry, enforced structurally by the `SEARCH:` branch in `src/main.rs` (no inner loop) and in the prompt by the `insist=true` suffix that `answer_prompt` in `src/lib.rs` appends on the second call.
- **Everything journaled.** Every step appends a JSON line to `journal.jsonl`
  (gitignored runtime artifact — never commit it).

## Commands

Tooling: `cargo install --locked just cargo-deny` (one-time).

| Task | Command |
|---|---|
| Format | `just fmt` |
| Lint (warnings are errors) | `just lint` |
| Type-check | `just check` |
| Dependency audit | `just deny` |
| Everything CI runs | `just ci` |
| Ask a question | `just run "your question"` |

Tests live in `tests/` as integration tests against the helpers in
`src/lib.rs`. CI runs `cargo test --locked` on every push and PR.

## Conventions

- **Lints** are configured in [Cargo.toml](Cargo.toml) `[lints]`: clippy
  `all` + `pedantic` as warnings, `unsafe_code = "forbid"`. CI runs clippy with
  `-D warnings`, so code must be warning-free. Prefer fixing a pedantic warning
  over `#[allow]`-ing it; if a lint is genuinely noise, add the allow in the
  `[lints.clippy]` table with a comment, not inline.
- **Formatting** is rustfmt defaults plus [rustfmt.toml](rustfmt.toml) (stable
  options only — unstable options break `cargo fmt` on the pinned stable
  toolchain). Always run `just fmt` before committing.
- **Toolchain** is pinned to an exact Rust version in
  [rust-toolchain.toml](rust-toolchain.toml) — the single source of truth that
  both local rustup and CI honor. Bump it there (nowhere else), and verify with
  `just ci`.
- **Dependencies**: `Cargo.lock` is committed (binary crate). New deps must
  pass `just deny` — if a new license appears, extend the allowlist in
  [deny.toml](deny.toml) with exactly that license, nothing broader.
- **Comment style**: `main.rs` uses section-banner comments tied to the
  numbered flow in the README. Keep new code consistent with that structure and
  update the README table if the flow changes.

## Configuration & secrets

Runtime config comes from `.env` at the repo root (loaded by `dotenvy`):
`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `FIRECRAWL_API_KEY`.

- `.env` is gitignored and contains real keys — never commit, print, or copy
  its contents. Keep [.env.example](.env.example) in sync when adding variables.
- A real `cargo run -- "question"` makes billed API calls (OpenRouter +
  Firecrawl). Use it sparingly as a final smoke test, not in loops.

## Before you're done

1. `just ci` passes (fmt-check, clippy, check, deny).
2. If behavior changed: one smoke run, e.g. `just run "what is the capital of France?"`.
3. README updated if the flow, knobs, or env vars changed.

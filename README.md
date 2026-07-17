# OpenRouter Chat Agent — Rust

Simple multi-turn terminal chat agent using OpenRouter's Chat Completions API.

## Prerequisites

- Rust 1.75+
- An [OpenRouter](https://openrouter.ai/) API key

## Setup

Edit `.env` at the repo root with your credentials:

```
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=openrouter/auto
```

`OPENROUTER_MODEL` is optional (defaults to `openrouter/auto`).

## Run

```bash
cd rust
cargo run
```

Type `exit` or `quit` to end a conversation.

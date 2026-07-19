# Development tasks. One-time setup:
#   cargo install --locked just cargo-deny

# List available recipes.
default:
    @just --list

# Format all Rust code.
fmt:
    cargo fmt --all

# Check formatting without modifying files.
fmt-check:
    cargo fmt --all --check

# Lint with clippy; warnings are errors, matching CI.
lint:
    cargo clippy --all-targets -- -D warnings

# Type-check against the committed lockfile.
check:
    cargo check --locked

# Audit dependencies for advisories, license issues, and duplicates.
deny:
    cargo deny check

# Run tests against the committed lockfile.
test:
    cargo test --locked

# Ask a question, e.g.:  just run "what is rust?"
run question:
    cargo run -- "{{question}}"

# Run coverage (needs cargo-llvm-cov installed; reports lib.rs + extracted helpers).
coverage:
    cargo llvm-cov --all-targets

# Everything CI runs, locally.
ci: fmt-check lint check test deny

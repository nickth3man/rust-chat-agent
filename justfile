# Development tasks. One-time setup:
#   cargo install --locked just cargo-deny cargo-llvm-cov
# The pinned toolchain (rust-toolchain.toml) includes llvm-tools-preview
# for cargo-llvm-cov; rustup installs it with `rustup show`.

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

# Coverage: line + region on the pinned stable toolchain (authoritative gate).
# Needs cargo-llvm-cov + llvm-tools-preview. Fails under 100% line or region.
coverage:
    cargo llvm-cov --locked --fail-under-lines 100 --fail-under-regions 100

# Coverage with text report (and HTML under target/llvm-cov/html).
coverage-report:
    cargo llvm-cov --locked --html --text --fail-under-lines 100 --fail-under-regions 100
    @echo "HTML report: target/llvm-cov/html/index.html"

# Coverage JSON summary for scripting (written to target/llvm-cov-summary.json).
coverage-json:
    cargo llvm-cov --locked --json --summary-only --fail-under-lines 100 --fail-under-regions 100 --output-path target/llvm-cov-summary.json

# Branch coverage (supplemental). Requires a nightly toolchain:
#   rustup toolchain install nightly
# Not part of `just ci`; branch instrumentation needs -Z flags unavailable on stable.
coverage-branch:
    cargo +nightly llvm-cov --locked --branch --fail-under-lines 100 --fail-under-regions 100

# Everything CI runs, locally. (Coverage is a separate local gate — not in CI.)
ci: fmt-check lint check test deny

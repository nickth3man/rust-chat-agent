// answerbot — research agent CLI.
//
// All orchestration (env loading, LLM/Firecrawl calls, journaling, printing)
// lives in the library crate (`src/lib.rs` and its `run` module) so it is
// testable against wiremock servers without billed API calls. This binary is
// intentionally a one-line wrapper. See AGENTS.md and README.md for the flow.

fn main() -> anyhow::Result<()> {
    answerbot::run_cli()
}

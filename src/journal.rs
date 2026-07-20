// The journal: one JSON line per event, appended to the path the caller
// supplies (never read here — see `AGENTS.md`: "everything journaled"). This
// single file is the audit trail, the dedup record, and the citation
// registry's paper trail all at once.
//
// We use `std::fs` (synchronous) rather than `tokio::fs`: these are tiny
// files (a few hundred bytes each) written a handful of times per run. The
// blocking duration is negligible compared to the network calls that
// dominate the runtime, and `tokio::fs` would add noise without measurable
// benefit. The tokio runtime is multi-threaded, so these brief synchronous
// calls do not stall other tasks.

use serde_json::{json, Value};
use std::io::Write;
use std::time::SystemTime;

/// Seconds since the Unix epoch, saturating to 0 for a `SystemTime` before
/// the epoch (clock skew / test fixtures) instead of panicking or erroring.
pub fn unix_ts(now: SystemTime) -> u64 {
    match now.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => d.as_secs(),
        Err(_) => 0,
    }
}

/// Write one journal line to an already-open sink. Separated from path open
/// so tests can exercise the write-failure warning without a full disk.
/// Takes `&mut dyn Write` (not `impl Write`) so Ok/Err coverage is not split
/// across monomorphizations that llvm-cov counts as separate regions.
pub fn write_journal_line(w: &mut dyn Write, path_for_msg: &str, line: &str) {
    match writeln!(w, "{line}") {
        Ok(()) => {}
        Err(e) => eprintln!("warning: journal write failed ({path_for_msg}): {e}"),
    }
}

/// Append `event` (with a `ts` field set to `ts`) as one JSON line to `path`.
/// Open/write failures are surfaced on stderr so silent journal loss is at
/// least observable to the operator; the run itself is never aborted for it.
pub fn journal_event(path: &str, ts: u64, mut event: Value) {
    event["ts"] = json!(ts);
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(mut f) => write_journal_line(&mut f, path, &event.to_string()),
        Err(e) => eprintln!("warning: could not open {path}: {e}"),
    }
}

/// `journal_event` using the current wall-clock time.
pub fn journal(path: &str, event: Value) {
    journal_event(path, unix_ts(SystemTime::now()), event);
}

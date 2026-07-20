// Tests for the retry-helper pure arithmetic in src/lib.rs:
//   - backoff_ms: exponential schedule capped at 4000ms
//   - is_retryable_status: HTTP status classification for network-retry policy
//
// These mirror the inline retry loops in src/main.rs (post_json, llm,
// rewrite_llm), but the arithmetic and the status classifier live in the
// pure lib so they can be exercised without standing up the binary.

use answerbot::{backoff_ms, is_retryable_status};

// -- backoff_ms: exponential schedule -------------------------------------

#[test]
fn backoff_ms_attempt_zero_is_250() {
    assert_eq!(backoff_ms(0), 250);
}

#[test]
fn backoff_ms_attempt_one_is_500() {
    assert_eq!(backoff_ms(1), 500);
}

#[test]
fn backoff_ms_attempt_two_is_1000() {
    assert_eq!(backoff_ms(2), 1000);
}

#[test]
fn backoff_ms_attempt_three_is_2000() {
    assert_eq!(backoff_ms(3), 2000);
}

#[test]
fn backoff_ms_attempt_four_is_4000() {
    assert_eq!(backoff_ms(4), 4000);
}

#[test]
fn backoff_ms_attempt_five_caps_at_4000() {
    assert_eq!(backoff_ms(5), 4000);
}

#[test]
fn backoff_ms_huge_attempt_caps_at_4000() {
    assert_eq!(backoff_ms(100), 4000);
}

#[test]
fn backoff_ms_attempt_max_u32_caps_at_4000() {
    assert_eq!(backoff_ms(u32::MAX), 4000);
}

// -- is_retryable_status: HTTP status classification ----------------------

#[test]
fn is_retryable_status_429_is_retryable() {
    assert!(is_retryable_status(429));
}

#[test]
fn is_retryable_status_500_is_retryable() {
    assert!(is_retryable_status(500));
}

#[test]
fn is_retryable_status_502_is_retryable() {
    assert!(is_retryable_status(502));
}

#[test]
fn is_retryable_status_503_is_retryable() {
    assert!(is_retryable_status(503));
}

#[test]
fn is_retryable_status_504_is_retryable() {
    assert!(is_retryable_status(504));
}

#[test]
fn is_retryable_status_599_upper_bound_is_retryable() {
    assert!(is_retryable_status(599));
}

#[test]
fn is_retryable_status_200_is_not_retryable() {
    assert!(!is_retryable_status(200));
}

#[test]
fn is_retryable_status_301_is_not_retryable() {
    assert!(!is_retryable_status(301));
}

#[test]
fn is_retryable_status_400_is_not_retryable() {
    assert!(!is_retryable_status(400));
}

#[test]
fn is_retryable_status_401_is_not_retryable() {
    assert!(!is_retryable_status(401));
}

#[test]
fn is_retryable_status_403_is_not_retryable() {
    assert!(!is_retryable_status(403));
}

#[test]
fn is_retryable_status_404_is_not_retryable() {
    assert!(!is_retryable_status(404));
}

#[test]
fn is_retryable_status_418_is_not_retryable() {
    assert!(!is_retryable_status(418));
}

#[test]
fn is_retryable_status_451_is_not_retryable() {
    assert!(!is_retryable_status(451));
}

#[test]
fn is_retryable_status_428_is_not_retryable() {
    // 428 (Precondition Required) is a 4xx — not in the retryable set.
    assert!(!is_retryable_status(428));
}

#[test]
fn is_retryable_status_600_outside_5xx_is_not_retryable() {
    // 600 is outside the 5xx range (500..=599) — not retryable per policy.
    assert!(!is_retryable_status(600));
}

#[test]
fn is_retryable_status_499_just_below_5xx_is_not_retryable() {
    assert!(!is_retryable_status(499));
}

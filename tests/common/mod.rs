// Shared test fixtures for the integration tests under tests/.
// Loaded via `mod common;` from each sibling .rs file — NOT compiled as a
// separate test target.

use answerbot::Source;

/// Minimal Source with only the `id` set. Use for tests that only care about
/// ID handling (citation validation, registry lookup, ID generation).
pub fn src(id: &str) -> Source {
    Source {
        id: id.into(),
        url: String::new(),
        title: String::new(),
        content: String::new(),
    }
}

/// Fully-populated Source. Use when the test asserts on title/url/content.
pub fn full_src(id: &str, title: &str, url: &str, content: &str) -> Source {
    Source {
        id: id.into(),
        title: title.into(),
        url: url.into(),
        content: content.into(),
    }
}

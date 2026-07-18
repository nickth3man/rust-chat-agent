//! Search adapters.  `BackendHttp` is deliberately shared by every adapter.
pub mod backends;
pub mod http;
pub mod registry;

pub use http::BackendHttp;
pub use registry::BackendRegistry;

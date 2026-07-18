//! Stable application contracts shared by the future search, agent, and session layers.

pub mod error;
pub mod provenance;
pub mod session;
pub mod types;

pub use error::{AppError, ToolNetError};
pub use provenance::{EvidenceEntry, TurnProvenance};
pub use types::{BackendKind, Category, SearchBackend, SearchHit};

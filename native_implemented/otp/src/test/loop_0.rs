#[cfg(feature = "runtime_full")]
mod runtime_full;
#[cfg(feature = "runtime_full")]
pub use runtime_full::*;

pub use super::module;
use super::module_id;

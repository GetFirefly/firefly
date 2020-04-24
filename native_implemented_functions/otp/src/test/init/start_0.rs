#[cfg(feature = "runtime_minimal")]
mod runtime_minimal;
#[cfg(feature = "runtime_minimal")]
pub use runtime_minimal::*;

#[cfg(feature = "runtime_full")]
mod runtime_full;
#[cfg(feature = "runtime_full")]
pub use runtime_full::*;

use super::{module, module_id};

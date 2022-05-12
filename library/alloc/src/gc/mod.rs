mod boxed;

pub use self::boxed::*;

use core::fmt;

/// Represents the types of errors that can occur during garbage collection.
///
/// See the documentation for each variant to get general advice for how to
/// handle these errors
#[derive(Debug, PartialEq, Eq)]
pub enum GcError {
    /// The system is out of memory, and there is not much you can do
    /// but panic, however this choice is left up to the caller
    AllocError,
    /// Occurs when a process is configured with a maximum heap size,
    /// and a projected heap growth is found to exceed the limit. In
    /// this situation the only meaningful thing to do is to kill the
    /// process
    MaxHeapSizeExceeded,
    /// Indicates that an allocation could not be filled without first
    /// performing a full sweep collection
    FullsweepRequired,
}
impl fmt::Display for GcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::AllocError => f.write_str("unable to allocate memory for garbage collection"),
            Self::MaxHeapSizeExceeded => f.write_str("maximum heap size exceeded"),
            Self::FullsweepRequired => f.write_str("a full garbage collection sweep is required"),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for GcError {}

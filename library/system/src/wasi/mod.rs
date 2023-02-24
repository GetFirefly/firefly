pub mod alloc;

pub use std::time;

/// Returns the configured page size
#[inline]
pub const fn page_size() -> usize {
    // Page size is always constant in WebAssembly, per the spec
    64 * 1024
}

/// Returns the number of logical CPUs online
#[inline]
pub const fn num_cpus() -> usize {
    1
}

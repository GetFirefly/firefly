pub mod spec;

pub use self::spec::*;

const HOST_TRIPLE: &'static str = env!("TARGET");

/// Get the host triple out of the build environment. This ensures that our
/// idea of the host triple is the same as for the set of libraries we've
/// actually built.  We can't just take LLVM's host triple because they
/// normalize all ix86 architectures to i386.
///
/// Instead of grabbing the host triple (for the current host), we grab (at
/// compile time) the target triple that the compiler is built with and
/// calling that (at runtime) the host triple.
pub fn host_triple() -> &'static str {
    HOST_TRIPLE
}

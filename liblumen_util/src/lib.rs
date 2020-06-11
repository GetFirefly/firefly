#![feature(optin_builtin_traits)]
// Used to prevent panic!(FatalError)
#![feature(negative_impls)]

pub mod diagnostics;
pub mod error;
pub mod ffi;
pub mod fs;
pub mod mem;
pub mod seq;
pub mod threading;
pub mod time;

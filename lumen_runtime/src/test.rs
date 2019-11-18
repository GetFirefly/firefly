pub mod r#loop;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest,
// so disable property-based tests and associated helpers completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod proptest;
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod strategy;
#[cfg(all(not(target_arch = "wasm32"), test))]
pub use self::proptest::*;

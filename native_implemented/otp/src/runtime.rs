#[cfg(all(feature = "runtime_minimal", feature = "runtime_full"))]
compile_error!("Only one runtime can be selected.");

#[cfg(feature = "runtime_minimal")]
pub use lumen_rt_minimal::*;

#[cfg(feature = "runtime_full")]
pub use lumen_rt_full::*;

#[cfg(not(any(feature = "runtime_minimal", feature = "runtime_full")))]
compile_error!(
    "One runtime must be selected with \"runtime_minimal\" or \"runtime_full\" feature."
);

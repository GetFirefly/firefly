#[cfg(test)]
pub mod loop_0;
#[cfg(test)]
pub mod process;

#[cfg(test)]
use liblumen_alloc::erts::term::prelude::*;
#[cfg(test)]
use liblumen_alloc::erts::process::ffi::ProcessSignal;

pub use lumen_rt_core::test::*;

#[cfg(test)]
use lumen_rt_core::test::once;

#[cfg(test)]
fn module() -> Atom {
    Atom::from_str("test")
}

#[cfg(test)]
pub(crate) fn once_crate() {
    once(&[]);
}

#[cfg(test)]
#[export_name = "__lumen_process_signal"]
#[thread_local]
static mut PROCESS_SIGNAL: ProcessSignal = ProcessSignal::None;

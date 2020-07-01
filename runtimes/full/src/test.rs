#[cfg(test)]
pub mod loop_0;
#[cfg(test)]
pub mod process;

#[cfg(test)]
use liblumen_alloc::erts::term::prelude::*;

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

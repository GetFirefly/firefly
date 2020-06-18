pub mod loop_0;
pub mod process;

use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_core::test::once;

fn module() -> Atom {
    Atom::from_str("test")
}

pub(crate) fn once_crate() {
    once(&[]);
}

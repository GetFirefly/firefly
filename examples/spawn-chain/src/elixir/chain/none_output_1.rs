//! defp none_output(_text) do
//!   :ok
//! end

use std::ffi::c_void;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    process.export_closure(
        super::module(),
        function(),
        ARITY,
        Some(native as *const c_void),
    )
}

// Private

#[native_implemented::function(none_output/1)]
fn result(_text: Term) -> Term {
    atom!("ok")
}

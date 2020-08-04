//! defp none_output(_text) do
//!   :ok
//! end

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    process.export_closure(super::module(), function(), ARITY, CLOSURE_NATIVE)
}

// Private

#[native_implemented::function(Elixir.Chain:none_output/1)]
fn result(_text: Term) -> Term {
    atom!("ok")
}

use liblumen_alloc::erts::exception::Result;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

pub fn new(first: Term, last: Term, process: &Process) -> Result {
    if first.is_integer() & last.is_integer() {
        process
            .map_from_slice(&[
                (atom_unchecked("__struct__"), atom_unchecked("Elixir.Range")),
                (atom_unchecked("first"), first),
                (atom_unchecked("last"), last),
            ])
            .map_err(|alloc_err| alloc_err.into())
    } else {
        Err(liblumen_alloc::badarg!().into())
    }
}

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn new(first: Term, last: Term, process: &Process) -> exception::Result<Term> {
    if first.is_integer() & last.is_integer() {
        process
            .map_from_slice(&[
                (Atom::str_to_term("__struct__"), Atom::str_to_term("Elixir.Range")),
                (Atom::str_to_term("first"), first),
                (Atom::str_to_term("last"), last),
            ])
            .map_err(|alloc_err| alloc_err.into())
    } else {
        Err(liblumen_alloc::badarg!().into())
    }
}

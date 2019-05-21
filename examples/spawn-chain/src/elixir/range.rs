use lumen_runtime::atom::Existence::DoNotCare;
use lumen_runtime::exception::Result;
use lumen_runtime::process::Process;
use lumen_runtime::term::Term;

pub fn new(first: Term, last: Term, process: &Process) -> Result {
    if first.is_integer() & last.is_integer() {
        Ok(Term::slice_to_map(
            &[
                (
                    Term::str_to_atom("__struct__", DoNotCare).unwrap(),
                    Term::str_to_atom("Elixir.Range", DoNotCare).unwrap(),
                ),
                (Term::str_to_atom("first", DoNotCare).unwrap(), first),
                (Term::str_to_atom("last", DoNotCare).unwrap(), last),
            ],
            process,
        ))
    } else {
        Err(lumen_runtime::badarg!())
    }
}

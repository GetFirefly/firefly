use std::sync::Arc;

use crate::exception::Result;
use crate::otp::erlang;
use crate::process::Process;
use crate::term::Term;

// I have no idea how this would work in LLVM generated code, but for testing `spawn/3` this allows
// `crate::instructions::Instructions::Apply` to translate Terms to Rust functions.
pub fn apply(
    module: Term,
    function: Term,
    arguments: Vec<Term>,
    arc_process: &Arc<Process>,
) -> Result {
    let arity = arguments.len();

    match unsafe { module.atom_to_string() }.as_ref().as_ref() {
        "erlang" => match unsafe { function.atom_to_string() }.as_ref().as_ref() {
            "+" => match arity {
                1 => erlang::number_or_badarith_1(arguments[0]),
                _ => Err(undef!(
                    module,
                    function,
                    to_list(arguments, arc_process),
                    arc_process
                )),
            },
            "self" => match arity {
                0 => Ok(erlang::self_0(arc_process)),
                _ => Err(undef!(
                    module,
                    function,
                    to_list(arguments, arc_process),
                    arc_process
                )),
            },
            _ => Err(undef!(
                module,
                function,
                to_list(arguments, arc_process),
                arc_process
            )),
        },
        _ => Err(undef!(
            module,
            function,
            to_list(arguments, arc_process),
            arc_process
        )),
    }
}

fn to_list(arguments: Vec<Term>, process: &Process) -> Term {
    arguments
        .into_iter()
        .rfold(Term::EMPTY_LIST, |acc, term| Term::cons(term, acc, process))
}

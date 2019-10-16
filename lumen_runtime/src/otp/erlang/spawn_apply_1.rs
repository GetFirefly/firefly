use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use crate::otp::erlang::apply_2;
use crate::process::spawn::options::Options;
use crate::scheduler::Scheduler;

pub(in crate::otp::erlang) fn native(
    process: &Process,
    options: Options,
    function: Term,
) -> exception::Result {
    if function.is_function() {
        let arguments = &[function, Term::NIL];

        // The :badarity error is raised in the child process and not in the parent process, so the
        // child process must be running the equivalent of `apply(functon, [])`.
        Scheduler::spawn_code(
            process,
            options,
            apply_2::module(),
            apply_2::function(),
            arguments,
            apply_2::code,
        )
        .and_then(|spawned| spawned.to_term(process))
        .map_err(|alloc| alloc.into())
    } else {
        Err(badarg!().into())
    }
}

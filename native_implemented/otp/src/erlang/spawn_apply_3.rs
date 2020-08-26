use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::apply::arguments_term_to_vec;
use crate::runtime::process::spawn::options::Options;
use crate::runtime::scheduler::Scheduled;

pub(in crate::erlang) fn result(
    process: &Process,
    options: Options,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result<Term> {
    let module_atom = term_try_into_atom!(module)?;
    let function_atom = term_try_into_atom!(function)?;
    let argument_vec = arguments_term_to_vec(arguments)?;

    process
        .scheduler()
        .unwrap()
        .spawn_module_function_arguments(
            Some(process),
            module_atom,
            function_atom,
            argument_vec,
            options,
        )
        .map(|spawned| spawned.to_term(process))
        .map_err(From::from)
}

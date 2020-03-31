use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_full::process::spawn::options::Options;
use lumen_rt_full::scheduler::Scheduler;

pub(in crate::erlang) fn native(
    process: &Process,
    options: Options,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result<Term> {
    let module_atom = term_try_into_atom!(module)?;
    let function_atom = term_try_into_atom!(function)?;

    let args = arguments.decode()?;
    if args.is_proper_list() {
        Scheduler::spawn_apply_3(process, options, module_atom, function_atom, arguments)
            .and_then(|spawned| spawned.to_term(process))
            .map_err(|e| e.into())
    } else {
        Err(TypeError)
            .with_context(|| format!("arguments ({}) must be a proper list", arguments))
            .map_err(From::from)
    }
}

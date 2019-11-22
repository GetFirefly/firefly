use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::process::spawn::options::Options;
use crate::scheduler::Scheduler;

pub(in crate::otp::erlang) fn native(
    process: &Process,
    options: Options,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result<Term> {
    let module_atom: Atom = module
        .try_into()
        .with_context(|| format!("module ({}) must be an atom", module))?;
    let function_atom: Atom = function
        .try_into()
        .with_context(|| format!("function ({}) must be an atom", function))?;

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

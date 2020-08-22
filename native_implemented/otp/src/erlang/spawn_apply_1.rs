use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::process::spawn::options::Options;
use crate::runtime::scheduler::Scheduled;

pub(in crate::erlang) fn result(
    process: &Process,
    function: Term,
    options: Options,
) -> exception::Result<Term> {
    let boxed_closure: Boxed<Closure> = function
        .try_into()
        .with_context(|| format!("function ({}) is not a function", function))?;

    process
        .scheduler()
        .unwrap()
        .spawn_closure(Some(process), boxed_closure, options)
        .map(|spawned| spawned.to_term(process))
        .map_err(From::from)
}

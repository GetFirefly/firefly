use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::{self, apply_2};
use crate::runtime;
use crate::runtime::process::spawn::options::Options;

pub(in crate::erlang) fn result(
    process: &Process,
    options: Options,
    function: Term,
) -> exception::Result<Term> {
    let _: Boxed<Closure> = function
        .try_into()
        .with_context(|| format!("function ({}) is not a function", function))?;
    let arguments = &[function, Term::NIL];

    // The :badarity error is raised in the child process and not in the parent process, so the
    // child process must be running the equivalent of `apply(function, [])`.
    runtime::process::spawn::native(
        Some(process),
        options,
        erlang::module(),
        apply_2::function(),
        arguments,
        apply_2::NATIVE,
    )
    .and_then(|spawned| spawned.schedule_with_parent(process).to_term(process))
    .map_err(|e| e.into())
}

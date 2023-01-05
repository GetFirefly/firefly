use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Closure, Term};

use crate::runtime::process::spawn::options::Options;
use crate::runtime::scheduler::Scheduled;

pub(in crate::erlang) fn result(
    process: &Process,
    function: Term,
    options: Options,
) -> Result<Term, NonNull<ErlangException>> {
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

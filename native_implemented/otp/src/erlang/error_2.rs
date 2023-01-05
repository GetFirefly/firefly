use std::ptr::NonNull;

use anyhow::*;
use firefly_rt::backtrace::Trace;

use firefly_rt::error::ErlangException;
use firefly_rt::function::ErlangResult;
use firefly_rt::term::{atoms, Term};

#[native_implemented::function(erlang:error/2)]
pub fn result(reason: Term, arguments: Term) -> Result<Term, NonNull<ErlangException>> {
    let err = ErlangException::new_with_meta(atoms::Error, reason, arguments, Trace::capture());
    Err(unsafe { NonNull::new_unchecked(Box::into_raw(err)) })
}

use std::ptr::NonNull;
use firefly_rt::backtrace::Trace;

use firefly_rt::error::ErlangException;
use firefly_rt::term::{atoms, Term};

#[native_implemented::function(erlang:exit/1)]
fn result(reason: Term) -> Result<Term, NonNull<ErlangException>> {
    let err = ErlangException::new(atoms::Exit, reason.into(), Trace::capture());
    Err(unsafe { NonNull::new_unchecked(Box::into_raw(err)) })
}

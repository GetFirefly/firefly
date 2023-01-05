use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

pub fn flag_is_not_a_supported_atom(flag: Term) -> Result<Term, NonNull<ErlangException>> {
    Err(anyhow!("flag ({}) is not a supported atom (label, monotonic_timestamp, print, receive, send, serial, spawn, strict_monotonic_timestamp, or timestamp)", flag).into())
}

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

pub fn flag_is_not_a_supported_atom(flag: Term) -> exception::Result<Term> {
    Err(anyhow!("flag ({}) is not a supported atom (label, monotonic_timestamp, print, receive, send, serial, spawn, strict_monotonic_timestamp, or timestamp)", flag).into())
}

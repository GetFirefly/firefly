use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use super::seq_trace::flag_is_not_a_supported_atom;

// See https://github.com/lumen/otp/blob/30e2bfb9f1fd5c65bd7d9a4159f88cdcf72023fa/erts/emulator/beam/erl_bif_trace.c#L1855-L1917
#[native_implemented::function(erlang:seq_trace_info/1)]
pub fn result(process: &Process, flag: Term) -> Result<Term, NonNull<ErlangException>> {
    let flag_name = term_try_into_atom!(flag)?.as_str();

    // Stub as if seq tracing is ALWAYS NOT enabled
    // See https://github.com/lumen/otp/blob/30e2bfb9f1fd5c65bd7d9a4159f88cdcf72023fa/erts/emulator/beam/erl_bif_trace.c#L1865-L1879
    match flag_name {
        "label" => Ok(label(process, flag)),
        "monotonic_timestamp"
        | "print"
        | "receive"
        | "send"
        | "spawn"
        | "strict_monotonic_timestamp"
        | "timestamp" => Ok(boolean_item(process, flag)),
        "serial" => Ok(serial(process, flag)),
        _ => flag_is_not_a_supported_atom(flag),
    }
}

fn boolean_item(process: &Process, item: Term) -> Term {
    tagged(process, item, false.into())
}

fn label(process: &Process, item: Term) -> Term {
    tagged(process, item, Term::Nil)
}

fn serial(process: &Process, item: Term) -> Term {
    tagged(process, item, Term::Nil)
}

fn tagged(process: &Process, tag: Term, value: Term) -> Term {
    process.tuple_term_from_term_slice(&[tag, value])
}

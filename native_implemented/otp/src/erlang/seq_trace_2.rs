use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use super::seq_trace::flag_is_not_a_supported_atom;

// See https://github.com/lumen/otp/blob/30e2bfb9f1fd5c65bd7d9a4159f88cdcf72023fa/erts/emulator/beam/erl_bif_trace.c#L1705-L1828
#[native_implemented::function(erlang:seq_trace/2)]
pub fn result(flag: Term, _value: Term) -> exception::Result<Term> {
    let flag_name = term_try_into_atom!(flag)?.name();

    match flag_name {
        "label" => unimplemented!(),
        "monotonic_timestamp" => unimplemented!(),
        "print" => unimplemented!(),
        "receive" => unimplemented!(),
        "send" => unimplemented!(),
        "serial" => unimplemented!(),
        "spawn" => unimplemented!(),
        "strict_monotonic_timestamp" => unimplemented!(),
        "timestamp" => unimplemented!(),
        _ => flag_is_not_a_supported_atom(flag),
    }
}

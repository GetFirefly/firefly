use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;
use firefly_rt::term::atoms;

use super::seq_trace::flag_is_not_a_supported_atom;

// See https://github.com/lumen/otp/blob/30e2bfb9f1fd5c65bd7d9a4159f88cdcf72023fa/erts/emulator/beam/erl_bif_trace.c#L1705-L1828
#[native_implemented::function(erlang:seq_trace/2)]
pub fn result(flag: Term, _value: Term) -> Result<Term, NonNull<ErlangException>> {
    let flag_name = term_try_into_atom!(flag)?;

    match flag_name {
        atoms::Label => unimplemented!(),
        atoms::MonotonicTimestamp => unimplemented!(),
        atoms::Print => unimplemented!(),
        atoms::Receive => unimplemented!(),
        atoms::Send => unimplemented!(),
        atoms::Serial => unimplemented!(),
        atoms::Spawn => unimplemented!(),
        atoms::StrictMonotonicTimestamp => unimplemented!(),
        atoms::Timestamp => unimplemented!(),
        _ => flag_is_not_a_supported_atom(flag),
    }
}

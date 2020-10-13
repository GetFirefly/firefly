use liblumen_alloc::erts::term::prelude::*;

// See https://github.com/lumen/otp/blob/30e2bfb9f1fd5c65bd7d9a4159f88cdcf72023fa/erts/emulator/beam/erl_bif_trace.c#L1938-L1957
#[native_implemented::function(erlang:seq_trace_print/2)]
pub fn result(_label: Term, _message: Term) -> Term {
    // Stub as if seq_trace token is always not set
    // See https://github.com/lumen/otp/blob/30e2bfb9f1fd5c65bd7d9a4159f88cdcf72023fa/erts/emulator/beam/erl_bif_trace.c#L1948-L1950
    false.into()
}

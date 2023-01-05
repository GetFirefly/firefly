use firefly_rt::term::Term;

// See https://github.com/lumen/otp/blob/30e2bfb9f1fd5c65bd7d9a4159f88cdcf72023fa/erts/emulator/beam/erl_bif_trace.c#L1919-L1936
#[native_implemented::function(erlang:seq_trace_print/1)]
pub fn result(_message: Term) -> Term {
    // Stub as if `tracing_token` is not set
    // See https://github.com/lumen/otp/blob/30e2bfb9f1fd5c65bd7d9a4159f88cdcf72023fa/erts/emulator/beam/erl_bif_trace.c#L1930
    false.into()
}

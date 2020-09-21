#[macro_use]
#[path = "test.rs"]
mod test;

#[path = "lib/erlang.rs"]
pub mod erlang;
#[path = "lib/maps.rs"]
pub mod maps;

test_stderr_substrings!(
    backtrace,
    vec![
        "native_implemented/otp/tests/lib/backtrace/init.erl:9, in init:bad_reverse/1",
        "native_implemented/otp/tests/lib/backtrace/init.erl:11, in init:bad_reverse/1",
        "native_implemented/otp/src/erlang/tl_1.rs:7, in erlang:tl/1",
        "Process (#PID<0.2.0>) exited abnormally.",
        "badarg"
    ]
);

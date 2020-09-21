//! Tests that only use `liblumen_otp` with itself and test Erlang source

#[macro_use]
#[path = "test.rs"]
mod test;

// extra layer so that all tests start with `lib::`
#[path = "internal/lib.rs"]
mod lib;

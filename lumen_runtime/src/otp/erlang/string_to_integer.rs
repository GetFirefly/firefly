use num_bigint::BigInt;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;

pub fn string_to_integer(process: &Process, string: &str) -> exception::Result {
    match BigInt::parse_bytes(string.as_bytes(), 10) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(badarg!().into()),
    }
}

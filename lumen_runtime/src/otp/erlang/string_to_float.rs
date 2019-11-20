use std::num::FpCategory;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn string_to_float(process: &Process, string: &str) -> exception::Result<Term> {
    match string.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) & !string.chars().any(|b| b == '.') {
                        Err(badarg!(process).into())
                    } else {
                        process.float(inner).map_err(|error| error.into())
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan | FpCategory::Infinite => Err(badarg!(process).into()),
                FpCategory::Zero => {
                    // Erlang does not track the difference without +0 and -0.
                    let zero = inner.abs();

                    process.float(zero).map_err(|error| error.into())
                }
            }
        }
        Err(_) => Err(badarg!(process).into()),
    }
}

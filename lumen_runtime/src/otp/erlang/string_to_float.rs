use std::num::FpCategory;

use anyhow::*;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn string_to_float(process: &Process, string: &str) -> InternalResult<Term> {
    match string.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) && !string.chars().any(|b| b == '.') {
                        Err(
                            anyhow!("float string ({}) does not contain decimal point", string)
                                .into(),
                        )
                    } else {
                        process.float(inner).map_err(|error| error.into())
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan => Err(anyhow!("Erlang does not support NANs ({})", string).into()),
                FpCategory::Infinite => {
                    Err(anyhow!("Erlang does not support infinities ({})", string).into())
                }
                FpCategory::Zero => {
                    if !string.chars().any(|b| b == '.') {
                        Err(
                            anyhow!("float string ({}) does not contain decimal point", string)
                                .into(),
                        )
                    } else {
                        // Erlang does not track the difference without +0 and -0.
                        let zero = inner.abs();

                        process.float(zero).map_err(From::from)
                    }
                }
            }
        }
        Err(error) => Err(error)
            .context(format!("string ({}) cannot be parsed as float", string))
            .map_err(From::from),
    }
}

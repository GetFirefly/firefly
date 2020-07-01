use std::num::FpCategory;

use anyhow::*;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::context;

pub fn string_to_float(
    process: &Process,
    name: &'static str,
    value: &str,
    quote: char,
) -> InternalResult<Term> {
    match value.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) && !value.chars().any(|b| b == '.') {
                        Err(anyhow!(
                            "{} does not contain decimal point",
                            context::string(name, quote, value)
                        )
                        .into())
                    } else {
                        process.float(inner).map_err(|error| error.into())
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan => Err(anyhow!("Erlang does not support NANs ({})", value).into()),
                FpCategory::Infinite => {
                    Err(anyhow!("Erlang does not support infinities ({})", value).into())
                }
                FpCategory::Zero => {
                    if !value.chars().any(|b| b == '.') {
                        Err(anyhow!(
                            "{} does not contain decimal point",
                            context::string(name, quote, value)
                        )
                        .into())
                    } else {
                        // Erlang does not track the difference without +0 and -0.
                        let zero = inner.abs();

                        process.float(zero).map_err(From::from)
                    }
                }
            }
        }
        Err(error) => Err(error)
            .context(format!(
                "{} cannot be parsed as float",
                context::string(name, quote, value),
            ))
            .map_err(From::from),
    }
}

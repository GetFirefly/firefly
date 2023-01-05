use std::num::FpCategory;

use anyhow::*;

use firefly_rt::process::Process;
use firefly_rt::*;
use firefly_rt::term::Term;

use crate::runtime::context;

pub fn string_to_float(
    process: &Process,
    name: &'static str,
    term: Term,
    string: &str,
) -> InternalResult<Term> {
    match string.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) && !string.chars().any(|b| b == '.') {
                        Err(anyhow!(
                            "{} does not contain decimal point",
                            context::string(name, term)
                        )
                        .into())
                    } else {
                        Ok(inner.into())
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan => Err(anyhow!("Erlang does not support NANs ({})", string).into()),
                FpCategory::Infinite => {
                    Err(anyhow!("Erlang does not support infinities ({})", string).into())
                }
                FpCategory::Zero => {
                    if !string.chars().any(|b| b == '.') {
                        Err(anyhow!(
                            "{} does not contain decimal point",
                            context::string(name, term)
                        )
                        .into())
                    } else {
                        // Erlang does not track the difference without +0 and -0.
                        let zero = inner.abs();

                        Ok(zero.into())
                    }
                }
            }
        }
        Err(error) => Err(error)
            .context(format!(
                "{} cannot be parsed as float",
                context::string(name, term),
            ))
            .map_err(From::from),
    }
}

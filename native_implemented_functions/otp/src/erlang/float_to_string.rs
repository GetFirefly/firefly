mod decimal_digits;
mod scientific_digits;

use std::convert::{TryFrom, TryInto};

use anyhow::Context;

use liblumen_alloc::erts::exception::{self, InternalResult};
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::proplist::TryPropListFromTermError;

use decimal_digits::DecimalDigits;
use scientific_digits::ScientificDigits;

pub fn float_to_string(float: Term, options: Options) -> exception::Result<String> {
    // `TryInto<f64> for Term` will convert integer terms to f64 too, which we don't want
    let float_f64: f64 = float_term_to_f64(float)?;

    let string = match options {
        // https://github.com/erlang/otp/blob/d293c3ff700c1a0992a32dc3da9ae18964893c23/erts/emulator/beam/bif.c#L3130-L3131
        Options::Decimals { digits, compact } => {
            //https://github.com/erlang/otp/blob/d293c3ff700c1a0992a32dc3da9ae18964893c23/erts/emulator/beam/bif.c#L3147-L3149
            float_to_decimal_string(float_f64, digits, compact)
        }
        // https://github.com/erlang/otp/blob/d293c3ff700c1a0992a32dc3da9ae18964893c23/erts/emulator/beam/bif.c#L3133-L3134
        Options::Scientific { digits } => {
            // https://github.com/erlang/otp/blob/d293c3ff700c1a0992a32dc3da9ae18964893c23/erts/emulator/beam/bif.c#L3151
            float_to_scientific_string(float_f64, digits)
        }
    };

    Ok(string)
}

pub enum Options {
    Decimals {
        digits: DecimalDigits,
        compact: bool,
    },
    Scientific {
        digits: ScientificDigits,
    },
}

impl Default for Options {
    fn default() -> Options {
        Options::Scientific {
            digits: Default::default(),
        }
    }
}

impl From<OptionsBuilder> for Options {
    fn from(options_builder: OptionsBuilder) -> Options {
        match options_builder.digits {
            Digits::None => Default::default(),
            Digits::Decimal(decimal_digits) => Options::Decimals {
                digits: decimal_digits,
                compact: options_builder.compact,
            },
            Digits::Scientific(scientific_digits) => Options::Scientific {
                digits: scientific_digits,
            },
        }
    }
}

impl TryFrom<Term> for Options {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let options_builder: OptionsBuilder = term.try_into()?;

        Ok(options_builder.into())
    }
}

// Private

fn float_term_to_f64(float_term: Term) -> InternalResult<f64> {
    match float_term.decode()? {
        TypedTerm::Float(float) => Ok(float.into()),
        _ => Err(TypeError)
            .context(format!("float ({}) is not a float", float_term))
            .map_err(From::from),
    }
}

fn float_to_decimal_string(f: f64, digits: DecimalDigits, compact: bool) -> String {
    let uncompacted = format!("{:0.*}", digits.into(), f);

    if compact {
        let parts: Vec<&str> = uncompacted.splitn(2, ".").collect();

        if parts.len() == 2 {
            let fractional_part_without_trailing_zeros = parts[1].trim_end_matches('0');

            format!(
                "{}.{:0<1}",
                parts[0], fractional_part_without_trailing_zeros
            )
        } else {
            uncompacted
        }
    } else {
        uncompacted
    }
}

fn float_to_scientific_string(f: f64, digits: ScientificDigits) -> String {
    // Rust does not use +/- or 2 digits with exponents, so need to modify the string after the
    // the fact, so we don't need to completely reimplement float formatting
    let rust_formatted = format!("{:0.*e}", digits.into(), f);
    let reverse_parts: Vec<&str> = rust_formatted.rsplitn(2, 'e').collect();
    assert_eq!(reverse_parts.len(), 2);
    let exponent = reverse_parts[0];
    let coefficient = reverse_parts[1];

    match exponent.chars().nth(0).unwrap() {
        '-' => format!("{}e-{:0>2}", coefficient, &exponent[1..]),
        '+' => format!("{}e+{:0>2}", coefficient, &exponent[1..]),
        _ => format!("{}e+{:0>2}", coefficient, exponent),
    }
}

enum Digits {
    None,
    Decimal(DecimalDigits),
    Scientific(ScientificDigits),
}

impl Default for Digits {
    fn default() -> Self {
        Self::None
    }
}

struct OptionsBuilder {
    compact: bool,
    digits: Digits,
}

impl OptionsBuilder {
    fn put_option_term(&mut self, option: Term) -> Result<&OptionsBuilder, anyhow::Error> {
        match option.decode().unwrap() {
            TypedTerm::Atom(atom) => match atom.name() {
                "compact" => {
                    self.compact = true;

                    Ok(self)
                }
                name => Err(TryAtomFromTermError(name)).context("supported atom option is compact"),
            },
            TypedTerm::Tuple(tuple) => {
                if tuple.len() == 2 {
                    let atom: Atom = tuple[0]
                        .try_into()
                        .map_err(|_| TryPropListFromTermError::KeywordKeyType)?;

                    match atom.name() {
                        "decimals" => {
                            let decimal_digits = tuple[1]
                                .try_into()
                                .context("decimals keyword option value")?;
                            self.digits = Digits::Decimal(decimal_digits);

                            Ok(self)
                        }
                        "scientific" => {
                            let scientific_digits = tuple[1]
                                .try_into()
                                .context("scientific keyword option value")?;
                            self.digits = Digits::Scientific(scientific_digits);

                            Ok(self)
                        }
                        name => Err(TryPropListFromTermError::KeywordKeyName(name).into()),
                    }
                } else {
                    Err(TryPropListFromTermError::TupleNotPair.into())
                }
            }
            _ => Err(TryPropListFromTermError::PropertyType.into()),
        }
    }
}

impl Default for OptionsBuilder {
    fn default() -> OptionsBuilder {
        OptionsBuilder {
            compact: false,
            digits: Default::default(),
        }
    }
}

const SUPPORTED_OPTIONS_CONTEXT: &str =
    "supported options are compact, {:decimal, 0..253}, or {:scientific, 0..249}";

impl TryFrom<Term> for OptionsBuilder {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let mut options_builder: OptionsBuilder = Default::default();
        let mut options_term = term;

        loop {
            match options_term.decode().unwrap() {
                TypedTerm::Nil => break,
                TypedTerm::List(cons) => {
                    options_builder
                        .put_option_term(cons.head)
                        .context(SUPPORTED_OPTIONS_CONTEXT)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(ImproperListError).context(SUPPORTED_OPTIONS_CONTEXT),
            }
        }

        Ok(options_builder)
    }
}

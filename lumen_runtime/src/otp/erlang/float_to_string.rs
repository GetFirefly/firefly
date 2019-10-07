use std::convert::{TryFrom, TryInto};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::term::{Atom, Term, TypedTerm};

pub fn float_to_string(float: Term, options: Options) -> Result<String, runtime::Exception> {
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

// > {decimals, Decimals :: 0..253}
pub struct DecimalDigits(u8);

impl DecimalDigits {
    const MAX_U8: u8 = 253;
}

impl Into<usize> for DecimalDigits {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<Term> for DecimalDigits {
    type Error = runtime::Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let decimal_digits_u8: u8 = term.try_into()?;

        if decimal_digits_u8 <= Self::MAX_U8 {
            Ok(Self(decimal_digits_u8))
        } else {
            Err(badarg!())
        }
    }
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

pub struct ScientificDigits(u8);

impl ScientificDigits {
    // > {scientific, Decimals :: 0..249}
    const MAX_U8: u8 = 249;
}

impl Default for ScientificDigits {
    fn default() -> Self {
        // > [float_binary(float) is the] same as float_to_binary(Float,[{scientific,20}]).
        Self(20)
    }
}

impl Into<usize> for ScientificDigits {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<Term> for ScientificDigits {
    type Error = runtime::Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let scientific_digits_u8: u8 = term.try_into()?;

        if scientific_digits_u8 <= Self::MAX_U8 {
            Ok(Self(scientific_digits_u8))
        } else {
            Err(badarg!())
        }
    }
}

// Private

fn float_term_to_f64(float_term: Term) -> Result<f64, runtime::Exception> {
    match float_term.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::Float(float) => Ok(float.into()),
            _ => Err(badarg!()),
        },
        _ => Err(badarg!()),
    }
}

fn float_to_decimal_string(f: f64, digits: DecimalDigits, _compact: bool) -> String {
    format!("{:0.*e}", digits.into(), f)
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
    fn put_option_term(&mut self, option: Term) -> Result<&OptionsBuilder, runtime::Exception> {
        match option.to_typed_term().unwrap() {
            TypedTerm::Atom(atom) => match atom.name() {
                "compact" => {
                    self.compact = true;

                    Ok(self)
                }
                _ => Err(badarg!()),
            },
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::Tuple(tuple) => {
                    if tuple.len() == 2 {
                        let atom: Atom = tuple[0].try_into()?;

                        match atom.name() {
                            "decimals" => {
                                self.digits = Digits::Decimal(tuple[1].try_into()?);

                                Ok(self)
                            }
                            "scientific" => {
                                self.digits = Digits::Scientific(tuple[1].try_into()?);

                                Ok(self)
                            }
                            _ => Err(badarg!()),
                        }
                    } else {
                        Err(badarg!())
                    }
                }
                _ => Err(badarg!()),
            },
            _ => Err(badarg!()),
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

impl TryFrom<Term> for OptionsBuilder {
    type Error = runtime::Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let mut options_builder: OptionsBuilder = Default::default();
        let mut options_term = term;

        loop {
            match options_term.to_typed_term().unwrap() {
                TypedTerm::Nil => break,
                TypedTerm::List(cons) => {
                    options_builder.put_option_term(cons.head)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(badarg!()),
            }
        }

        Ok(options_builder.into())
    }
}

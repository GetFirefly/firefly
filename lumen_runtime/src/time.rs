use std::convert::{TryFrom, TryInto};

use num_bigint::BigInt;
use num_traits::Zero;

use crate::exception::Exception;
use crate::integer::big;
use crate::term::{Tag::*, Term};

pub mod monotonic;

pub fn convert(time: BigInt, from_unit: Unit, to_unit: Unit) -> BigInt {
    if from_unit == to_unit {
        time
    } else {
        let from_hertz = from_unit.hertz();
        let to_hertz = to_unit.hertz();

        if from_hertz <= to_hertz {
            time * ((to_hertz / from_hertz) as i32)
        } else {
            // mimic behavior of erts_napi_convert_time_unit, so that rounding is the same
            let denominator: BigInt = (from_hertz / to_hertz).into();
            let zero: BigInt = Zero::zero();

            if zero <= time {
                time / denominator
            } else {
                (time - (denominator.clone() - 1)) / denominator
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Hertz(usize),
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    Native,
    PerformanceCounter,
}

impl Unit {
    const MILLISECOND_HERTZ: usize = 1_000;

    pub fn hertz(&self) -> usize {
        match self {
            Unit::Hertz(hertz) => *hertz,
            Unit::Second => 1,
            Unit::Millisecond => Self::MILLISECOND_HERTZ,
            Unit::Microsecond => 1_000_000,
            Unit::Nanosecond => 1_000_000_000,
            // As a side-channel protection browsers limit most counters to 1 millisecond resolution
            Unit::Native => Self::MILLISECOND_HERTZ,
            Unit::PerformanceCounter => Self::MILLISECOND_HERTZ,
        }
    }
}

impl TryFrom<Term> for Unit {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Unit, Exception> {
        match term.tag() {
            SmallInteger => {
                let hertz: usize = term.try_into()?;

                Ok(Unit::Hertz(hertz))
            }
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    BigInteger => {
                        let big_integer: &big::Integer = term.unbox_reference();
                        let big_integer_usize: usize = big_integer.try_into()?;

                        Ok(Unit::Hertz(big_integer_usize))
                    }
                    _ => Err(badarg!()),
                }
            }
            Atom => {
                let term_string = unsafe { term.atom_to_string() };
                let mut result = Err(badarg!());

                for (s, unit) in [
                    ("second", Unit::Second),
                    ("seconds", Unit::Second),
                    ("millisecond", Unit::Millisecond),
                    ("milli_seconds", Unit::Millisecond),
                    ("microsecond", Unit::Microsecond),
                    ("micro_seconds", Unit::Microsecond),
                    ("nanosecond", Unit::Nanosecond),
                    ("nano_seconds", Unit::Nanosecond),
                    ("native", Unit::Native),
                    ("perf_counter", Unit::PerformanceCounter),
                ]
                .iter()
                {
                    if term_string.as_ref() == s {
                        result = Ok(*unit);
                        break;
                    }
                }

                result
            }
            _ => Err(badarg!()),
        }
    }
}

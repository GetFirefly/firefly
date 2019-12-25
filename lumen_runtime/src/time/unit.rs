use std::convert::{TryFrom, TryInto};

use anyhow::*;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(test, derive(Debug))]
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

    pub fn to_term(&self, process: &Process) -> AllocResult<Term> {
        match self {
            Unit::Hertz(hertz) => process.integer(*hertz),
            Unit::Second => Ok(atom!("second")),
            Unit::Millisecond => Ok(atom!("millisecond")),
            Unit::Microsecond => Ok(atom!("microsecond")),
            Unit::Nanosecond => Ok(atom!("nanosecond")),
            Unit::Native => Ok(atom!("native")),
            Unit::PerformanceCounter => Ok(atom!("perf_counter")),
        }
    }
}

const NON_POSITIVE_HERTZ_CONTEXT: &str = "hertz must be positive";

impl TryFrom<Term> for Unit {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term.decode().unwrap() {
            TypedTerm::SmallInteger(small_integer) => {
                let hertz: usize = small_integer
                    .try_into()
                    .context(NON_POSITIVE_HERTZ_CONTEXT)?;

                if 0 < hertz {
                    Ok(Unit::Hertz(hertz))
                } else {
                    Err(TryIntoIntegerError::OutOfRange).context(NON_POSITIVE_HERTZ_CONTEXT)
                }
            }
            TypedTerm::BigInteger(big_integer) => {
                let big_integer_usize: usize =
                    big_integer.try_into().context(NON_POSITIVE_HERTZ_CONTEXT)?;

                Ok(Unit::Hertz(big_integer_usize))
            }
            TypedTerm::Atom(atom) => {
                let atom_name = atom.name();
                let mut option = None;

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
                    if &atom_name == s {
                        option = Some(*unit);
                        break;
                    }
                }

                match option {
                    Some(unit) => Ok(unit),
                    None => Err(TryAtomFromTermError(atom_name).into())
                }
            }
            _ => Err(TypeError.into()),
        }.context("supported units are :second, :seconds, :millisecond, :milli_seconds, :microsecond, :micro_seconds, :nanosecond, :nano_seconds, :native, :perf_counter, or hertz (positive integer)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::scheduler::with_process;

    #[test]
    fn zero_errors_hertz_must_be_positive() {
        with_process(|process| {
            let term: Term = process.integer(0).unwrap();

            let result: Result<Unit, _> = term.try_into();

            assert!(result.is_err());

            let formatted = format!("{:?}", result.unwrap_err());

            assert!(formatted.contains("hertz must be positive"));
            assert!(formatted.contains("supported units are :second, :seconds, :millisecond, :milli_seconds, :microsecond, :micro_seconds, :nanosecond, :nano_seconds, :native, :perf_counter, or hertz (positive integer)"));
        });
    }
}

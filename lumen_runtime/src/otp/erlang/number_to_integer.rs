use num_bigint::BigInt;

use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Float, SmallInteger, Term, TypedTerm};

pub enum NumberToInteger {
    NotANumber,
    Integer(Term),
    F64(f64),
}

impl From<Term> for NumberToInteger {
    fn from(number: Term) -> Self {
        match number.to_typed_term().unwrap() {
            TypedTerm::SmallInteger(_) => Self::Integer(number),
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(_) => Self::Integer(number),
                TypedTerm::Float(float) => Self::F64(float.into()),
                _ => Self::NotANumber,
            },
            _ => Self::NotANumber,
        }
    }
}

pub fn f64_to_integer(process: &Process, f: f64) -> Result<Term, Exception> {
    // skip creating a BigInt if f64 can fit in small integer.
    if (SmallInteger::MIN_VALUE as f64).max(Float::INTEGRAL_MIN) <= f
        && f <= (SmallInteger::MAX_VALUE as f64).min(Float::INTEGRAL_MAX)
    {
        process.integer(f as isize)
    } else {
        let string = f.to_string();
        let bytes = string.as_bytes();
        let big_int = BigInt::parse_bytes(bytes, 10).unwrap();

        process.integer(big_int)
    }
    .map_err(|alloc| alloc.into())
}

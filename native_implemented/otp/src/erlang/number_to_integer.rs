use num_bigint::BigInt;

use firefly_rt::process::Process;
use firefly_rt::term::{Float, Integer, Term};

pub enum NumberToInteger {
    NotANumber,
    Integer(Term),
    F64(f64),
}

impl From<Term> for NumberToInteger {
    fn from(number: Term) -> Self {
        match number {
            Term::Int(_) => Self::Integer(number),
            Term::BigInt(_) => Self::Integer(number),
            Term::Float(float) => Self::F64(float.into()),
            _ => Self::NotANumber,
        }
    }
}

pub fn f64_to_integer(process: &Process, f: f64) -> Term {
    // skip creating a BigInt if f64 can fit in small integer.
    if (Integer::MIN_SMALL as f64).max(Float::I64_LOWER_BOUNDARY) <= f
        && f <= (Integer::MAX_SMALL as f64).min(Float::I64_UPPER_BOUNDARY)
    {
        process.integer(f as isize).unwrap()
    } else {
        let string = f.to_string();
        let bytes = string.as_bytes();
        let big_int = BigInt::parse_bytes(bytes, 10).unwrap();

        process.integer(big_int).unwrap()
    }
}

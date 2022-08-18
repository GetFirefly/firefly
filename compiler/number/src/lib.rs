#![feature(let_else)]
#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate test;

mod bigint_to_float;
pub use bigint_to_float::bigint_to_double;

mod integer;
pub use integer::Integer;

mod float;
pub use float::{f16, Float, FloatError};

mod number;
pub use number::Number;

pub use num_bigint as bigint;
pub use num_bigint::{BigInt, Sign};
pub use num_traits as traits;
pub use num_traits::{cast, int::PrimInt, FromPrimitive, NumCast, ToPrimitive};

#[derive(Debug, Copy, Clone)]
pub struct DivisionError;

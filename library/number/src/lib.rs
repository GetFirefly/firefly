#![no_std]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;
#[cfg(test)]
extern crate test;

mod bigint_to_float;
pub use bigint_to_float::bigint_to_double;

mod integer;
pub use integer::*;

mod float;
pub use float::{f16, Float, FloatError};

mod number;
pub use number::Number;

pub use num_bigint as bigint;
pub use num_bigint::{BigInt, BigUint, Sign, ToBigInt, ToBigUint};
pub use num_traits as traits;
pub use num_traits::{cast, int::PrimInt, Num, NumCast, Pow};

/// This occurs when an invalid division is performed (i.e. by zero)
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DivisionError;

/// This occurs when an arithmetic operations arguments are invalid
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct InvalidArithmeticError;

/// This occurs when a shift operand is invalid/too large
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ShiftError;

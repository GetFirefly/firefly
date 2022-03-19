mod bigint_to_float;
pub use bigint_to_float::bigint_to_double;

mod integer;
pub use integer::Integer;

mod float;
pub use float::{Float, FloatError};

mod number;
pub use number::Number;

mod binary;

pub use num_bigint as bigint;
pub use num_bigint::BigInt;
pub use num_traits as traits;
pub use num_traits::{cast, FromPrimitive, NumCast, ToPrimitive};

#[derive(Debug, Copy, Clone)]
pub struct DivisionError;

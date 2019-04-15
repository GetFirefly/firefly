use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

use num_bigint::{BigInt, Sign::*};
use num_traits::Float;

use crate::exception::Exception;
use crate::term::{Tag::BigInteger, Term};

pub struct Integer {
    #[allow(dead_code)]
    header: Term,
    pub inner: BigInt,
}

impl Integer {
    pub fn new(inner: BigInt) -> Self {
        Integer {
            header: Term {
                tagged: BigInteger as usize,
            },
            inner,
        }
    }
}

impl Eq for Integer {}

impl From<Integer> for f64 {
    fn from(integer: Integer) -> f64 {
        (&integer).into()
    }
}

impl From<&Integer> for f64 {
    fn from(integer_ref: &Integer) -> f64 {
        let big_int = &integer_ref.inner;

        let (sign, bytes) = big_int.to_bytes_be();
        let unsigned_f64 = bytes
            .iter()
            .fold(0_f64, |acc, byte| 256.0 * acc + (*byte as f64));

        match sign {
            Minus => -1.0 * unsigned_f64,
            _ => unsigned_f64,
        }
    }
}

impl Hash for Integer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

impl PartialEq for Integer {
    fn eq(&self, other: &Integer) -> bool {
        self.inner == other.inner
    }
}

impl TryFrom<Integer> for usize {
    type Error = Exception;

    fn try_from(integer: Integer) -> Result<usize, Exception> {
        big_int_to_usize(&integer.inner)
    }
}

impl TryFrom<&Integer> for usize {
    type Error = Exception;

    fn try_from(integer_ref: &Integer) -> Result<usize, Exception> {
        big_int_to_usize(&integer_ref.inner)
    }
}

pub fn big_int_to_usize(big_int: &BigInt) -> Result<usize, Exception> {
    match big_int.sign() {
        Plus => {
            let (_, bytes) = big_int.to_bytes_be();
            let integer_usize = bytes
                .iter()
                .fold(0_usize, |acc, byte| (acc << 8) | (*byte as usize));

            Ok(integer_usize)
        }
        NoSign => Ok(0),
        Minus => Err(badarg!()),
    }
}

pub unsafe fn integral_f64_to_big_int(integral: f64) -> BigInt {
    let (mantissa, exponent, sign) = Float::integer_decode(integral);
    let mantissa_big_int: BigInt = mantissa.into();

    let scaled = if exponent < 0 {
        let right_shift = (-exponent) as usize;

        mantissa_big_int >> right_shift
    } else if exponent == 0 {
        mantissa_big_int
    } else {
        let left_shift = exponent as usize;

        mantissa_big_int << left_shift
    };

    sign * scaled
}

#[cfg(test)]
mod tests {
    use super::*;

    mod integral_f64_to_big_int {
        use super::*;

        mod with_negative_exponent {
            use super::*;

            #[test]
            fn with_negative_sign() {
                let f = -3.0;

                let (_mantissa, exponent, sign) = Float::integer_decode(f);

                assert!(exponent < 0);
                assert_eq!(sign, -1);

                assert_eq!(unsafe { integral_f64_to_big_int(f) }, BigInt::from(-3))
            }

            #[test]
            fn with_positive_sign() {
                let f = 3.0;

                let (_mantissa, exponent, sign) = Float::integer_decode(f);

                assert!(exponent < 0);
                assert_eq!(sign, 1);

                assert_eq!(unsafe { integral_f64_to_big_int(f) }, BigInt::from(3))
            }
        }

        mod with_zero_exponent {
            use super::*;

            #[test]
            fn with_negative_sign() {
                let f = -4503599627370496.0;

                let (_mantissa, exponent, sign) = Float::integer_decode(f);

                assert_eq!(exponent, 0);
                assert_eq!(sign, -1);

                assert_eq!(
                    unsafe { integral_f64_to_big_int(f) },
                    BigInt::from(-4503599627370496_i64)
                )
            }

            #[test]
            fn with_positive_sign() {
                let f = 4503599627370496.0;

                let (_mantissa, exponent, sign) = Float::integer_decode(f);

                assert_eq!(exponent, 0);
                assert_eq!(sign, 1);

                assert_eq!(
                    unsafe { integral_f64_to_big_int(f) },
                    BigInt::from(4503599627370496_u64)
                )
            }
        }

        mod with_positive_exponent {
            use super::*;

            #[test]
            fn with_negative_sign() {
                let f = -9007199254740992.0;

                let (_mantissa, exponent, sign) = Float::integer_decode(f);

                assert!(0 < exponent);
                assert_eq!(sign, -1);

                assert_eq!(
                    unsafe { integral_f64_to_big_int(f) },
                    BigInt::from(-9007199254740992_i64)
                )
            }

            #[test]
            fn with_positive_sign() {
                let f = 9007199254740992.0;

                let (_mantissa, exponent, sign) = Float::integer_decode(f);

                assert!(0 < exponent);
                assert_eq!(sign, 1);

                assert_eq!(
                    unsafe { integral_f64_to_big_int(f) },
                    BigInt::from(9007199254740992_u64)
                )
            }
        }
    }
}

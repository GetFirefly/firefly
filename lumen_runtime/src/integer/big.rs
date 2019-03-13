use std::convert::TryFrom;

use num_bigint::{BigInt, Sign::*};

use crate::bad_argument::BadArgument;
use crate::term::{Tag, Term};

pub struct Integer {
    #[allow(dead_code)]
    header: Term,
    pub inner: BigInt,
}

impl Integer {
    pub fn new(inner: BigInt) -> Self {
        Integer {
            header: Term {
                tagged: Tag::BigInteger as usize,
            },
            inner,
        }
    }
}

impl TryFrom<Integer> for usize {
    type Error = BadArgument;

    fn try_from(integer: Integer) -> Result<usize, BadArgument> {
        big_int_to_usize(&integer.inner)
    }
}

impl TryFrom<&Integer> for usize {
    type Error = BadArgument;

    fn try_from(integer_ref: &Integer) -> Result<usize, BadArgument> {
        big_int_to_usize(&integer_ref.inner)
    }
}

pub fn big_int_to_usize(big_int: &BigInt) -> Result<usize, BadArgument> {
    match big_int.sign() {
        Plus => {
            let (_, bytes) = big_int.to_bytes_be();
            let integer_usize = bytes
                .iter()
                .fold(0_usize, |acc, byte| (acc << 8) | (*byte as usize));

            Ok(integer_usize)
        }
        NoSign => Ok(0),
        Minus => Err(bad_argument!()),
    }
}

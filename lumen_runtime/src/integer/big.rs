use std::hash::{Hash, Hasher};

use num_bigint::{BigInt, Sign::*};

use crate::exception::Exception;
use crate::process::{Process, TryFromInProcess};
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

impl Hash for Integer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

impl PartialEq for Integer {
    fn eq(&self, other: &Integer) -> bool {
        self.inner == other.inner
    }

    fn ne(&self, other: &Integer) -> bool {
        !self.eq(other)
    }
}

impl TryFromInProcess<Integer> for usize {
    fn try_from_in_process(
        integer: Integer,
        mut process: &mut Process,
    ) -> Result<usize, Exception> {
        big_int_to_usize(&integer.inner, &mut process)
    }
}

impl TryFromInProcess<&Integer> for usize {
    fn try_from_in_process(
        integer_ref: &Integer,
        mut process: &mut Process,
    ) -> Result<usize, Exception> {
        big_int_to_usize(&integer_ref.inner, &mut process)
    }
}

pub fn big_int_to_usize(big_int: &BigInt, mut process: &mut Process) -> Result<usize, Exception> {
    match big_int.sign() {
        Plus => {
            let (_, bytes) = big_int.to_bytes_be();
            let integer_usize = bytes
                .iter()
                .fold(0_usize, |acc, byte| (acc << 8) | (*byte as usize));

            Ok(integer_usize)
        }
        NoSign => Ok(0),
        Minus => Err(bad_argument!(&mut process)),
    }
}

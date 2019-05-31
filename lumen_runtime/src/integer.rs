use std::cmp::Ordering::{self, Equal};
use std::convert::{TryFrom, TryInto};

use num_bigint::{BigInt, Sign};

use crate::exception::Exception;

pub mod big;
pub mod small;

use crate::integer::big::big_int_to_usize;

#[cfg_attr(test, derive(Debug))]
pub enum Integer {
    Small(small::Integer),
    Big(BigInt),
}

impl Eq for Integer {}

impl From<char> for Integer {
    fn from(c: char) -> Integer {
        (c as usize).into()
    }
}

impl From<i32> for Integer {
    fn from(i: i32) -> Integer {
        (i as isize).into()
    }
}

impl From<i64> for Integer {
    fn from(i: i64) -> Integer {
        if (small::MIN as i64) <= i && i <= (small::MAX as i64) {
            Integer::Small(small::Integer(i as isize))
        } else {
            Integer::Big(i.into())
        }
    }
}

impl From<isize> for Integer {
    fn from(i: isize) -> Integer {
        if small::MIN <= i && i <= small::MAX {
            Integer::Small(small::Integer(i))
        } else {
            Integer::Big(i.into())
        }
    }
}

impl From<u8> for Integer {
    fn from(u: u8) -> Integer {
        (u as usize).into()
    }
}

impl From<usize> for Integer {
    fn from(u: usize) -> Integer {
        if (u as isize) <= small::MAX {
            Integer::Small(small::Integer(u as isize))
        } else {
            Integer::Big(u.into())
        }
    }
}

impl From<BigInt> for Integer {
    fn from(big_int: BigInt) -> Integer {
        let small_min_big_int: BigInt = small::MIN.into();
        let small_max_big_int: BigInt = small::MAX.into();

        if (small_min_big_int <= big_int) & (big_int <= small_max_big_int) {
            let small_isize = big_int
                .to_signed_bytes_be()
                .iter()
                .fold(0_isize, |acc, byte| (acc << 8) | (*byte as isize));

            Integer::Small(small::Integer(small_isize))
        } else {
            Integer::Big(big_int)
        }
    }
}

impl Ord for Integer {
    fn cmp(&self, other: &Integer) -> Ordering {
        match (self, other) {
            (
                Integer::Small(small::Integer(self_isize)),
                Integer::Small(small::Integer(other_isize)),
            ) => self_isize.cmp(other_isize),
            (_, _) => unimplemented!(),
        }
    }
}

impl PartialEq for Integer {
    fn eq(&self, other: &Integer) -> bool {
        self.cmp(other) == Equal
    }
}

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Integer) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TryFrom<Integer> for usize {
    type Error = Exception;

    fn try_from(integer: Integer) -> Result<usize, Exception> {
        match integer {
            Integer::Small(small::Integer(untagged)) => untagged.try_into().map_err(|_| badarg!()),
            Integer::Big(big_int) => big_int_to_usize(&big_int),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_one() {
        let big_int: BigInt = (-1).into();

        let integer: Integer = big_int.into();

        assert_eq!(integer, Integer::Small(small::Integer(-1)));
    }
}

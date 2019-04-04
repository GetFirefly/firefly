use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug};

use num_bigint::BigInt;

use crate::exception::Exception;

pub mod big;
pub mod small;

use crate::integer::big::big_int_to_usize;

pub enum Integer {
    Small(small::Integer),
    Big(BigInt),
}

impl Debug for Integer {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Integer::Small(_) => unimplemented!(),
            Integer::Big(_) => unimplemented!(),
        }
    }
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
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq for Integer {
    fn eq(&self, other: &Integer) -> bool {
        match (self, other) {
            (
                Integer::Small(small::Integer(self_isize)),
                Integer::Small(small::Integer(other_isize)),
            ) => self_isize == other_isize,
            (_, _) => unimplemented!(),
        }
    }
}

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Integer) -> Option<Ordering> {
        match (self, other) {
            (
                Integer::Small(small::Integer(self_isize)),
                Integer::Small(small::Integer(other_isize)),
            ) => self_isize.partial_cmp(other_isize),
            (_, _) => unimplemented!(),
        }
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

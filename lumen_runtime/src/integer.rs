use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};

use num_bigint::BigInt;

use crate::process::{DebugInProcess, OrderInProcess, Process};
use crate::term::BadArgument;

pub mod big;
pub mod small;

use crate::integer::big::big_int_to_usize;

pub enum Integer {
    Small(small::Integer),
    Big(BigInt),
}

impl DebugInProcess for Integer {
    fn format_in_process(&self, _process: &Process) -> String {
        match self {
            Integer::Small(_) => unimplemented!(),
            Integer::Big(_) => unimplemented!(),
        }
    }
}

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

impl OrderInProcess for Integer {
    fn cmp_in_process(&self, other: &Self, _process: &Process) -> Ordering {
        match (self, other) {
            (
                Integer::Small(small::Integer(self_isize)),
                Integer::Small(small::Integer(other_isize)),
            ) => self_isize.cmp(other_isize),
            (_, _) => unimplemented!(),
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

impl TryFrom<Integer> for usize {
    type Error = BadArgument;

    fn try_from(integer: Integer) -> Result<usize, BadArgument> {
        match integer {
            Integer::Small(small::Integer(untagged)) => {
                untagged.try_into().map_err(|_| BadArgument)
            }
            Integer::Big(big_int) => big_int_to_usize(&big_int),
        }
    }
}

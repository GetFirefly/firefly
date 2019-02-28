use std::cmp::Ordering;

use crate::process::{DebugInProcess, OrderInProcess, Process};

pub mod big;
pub mod small;

pub enum Integer {
    Small(small::Integer),
    Big(rug::Integer),
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
            Integer::Big(rug::Integer::from(i))
        }
    }
}

impl From<usize> for Integer {
    fn from(u: usize) -> Integer {
        if (u as isize) <= small::MAX {
            Integer::Small(small::Integer(u as isize))
        } else {
            Integer::Big(rug::Integer::from(u))
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

impl From<Integer> for usize {
    fn from(integer: Integer) -> usize {
        match integer {
            Integer::Small(small::Integer(untagged)) => {
                if 0 <= untagged {
                    untagged as usize
                } else {
                    panic!(
                        "Small integer ({:?}) is less than 0 and cannot be converted to usize",
                        untagged
                    )
                }
            }
            Integer::Big(rug_integer) => rug_integer.to_usize().unwrap_or_else(|| {
                panic!("Big integer {:?} cannot be converted to usize", rug_integer)
            }),
        }
    }
}

impl From<rug::Integer> for Integer {
    fn from(rug_integer: rug::Integer) -> Integer {
        if (small::MIN <= rug_integer) & (rug_integer <= small::MAX) {
            Integer::Small(small::Integer(rug_integer.to_isize().unwrap()))
        } else {
            Integer::Big(rug_integer)
        }
    }
}

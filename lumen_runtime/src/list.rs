#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::{Eq, PartialEq};
use std::fmt::{self, Debug};

use crate::term::Term;

pub type List = *const Term;

/// A cons cell in a list
#[repr(C)]
pub struct Cons {
    head: Term,
    tail: Term,
}

impl Cons {
    pub fn new(head: Term, tail: Term) -> Cons {
        Cons { head, tail }
    }

    pub fn head(&self) -> Term {
        self.head
    }

    pub fn tail(&self) -> Term {
        self.tail
    }
}

impl Debug for Cons {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Cons::new({:?}. {:?})", self.head, self.tail)
    }
}

impl Eq for Cons {}

impl PartialEq for Cons {
    fn eq(&self, other: &Cons) -> bool {
        (self.head == other.head) & (self.tail == other.tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod eq {
        use super::*;

        #[test]
        fn with_proper() {
            let cons = Cons::new(0.into(), Term::EMPTY_LIST);
            let equal = Cons::new(0.into(), Term::EMPTY_LIST);
            let unequal = Cons::new(1.into(), Term::EMPTY_LIST);

            assert_eq!(cons, cons);
            assert_eq!(cons, equal);
            assert_ne!(cons, unequal);
        }

        #[test]
        fn with_improper() {
            let cons = Cons::new(0.into(), 1.into());
            let equal = Cons::new(0.into(), 1.into());
            let unequal = Cons::new(1.into(), 0.into());

            assert_eq!(cons, cons);
            assert_eq!(cons, equal);
            assert_ne!(cons, unequal);
        }
    }
}

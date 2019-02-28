#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering;

use crate::process::{DebugInProcess, OrderInProcess, Process};
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

impl DebugInProcess for Cons {
    fn format_in_process(&self, process: &Process) -> String {
        format!(
            "Cons::new({}, {})",
            self.head.format_in_process(process),
            self.tail.format_in_process(process)
        )
    }
}

impl OrderInProcess for Cons {
    fn cmp_in_process(&self, other: &Cons, process: &Process) -> Ordering {
        match self.head.cmp_in_process(&other.head, process) {
            Ordering::Equal => self.tail.cmp_in_process(&other.tail, process),
            ordering => ordering,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod eq {
        use super::*;

        use crate::process::IntoProcess;

        #[test]
        fn with_proper() {
            let mut process: Process = Default::default();
            let cons = Cons::new(0.into_process(&mut process), Term::EMPTY_LIST);
            let equal = Cons::new(0.into_process(&mut process), Term::EMPTY_LIST);
            let unequal = Cons::new(1.into_process(&mut process), Term::EMPTY_LIST);

            assert_eq_in_process!(cons, cons, process);
            assert_eq_in_process!(cons, equal, process);
            assert_ne_in_process!(cons, unequal, process);
        }

        #[test]
        fn with_improper() {
            let mut process: Process = Default::default();
            let cons = Cons::new(0.into_process(&mut process), 1.into_process(&mut process));
            let equal = Cons::new(0.into_process(&mut process), 1.into_process(&mut process));
            let unequal = Cons::new(1.into_process(&mut process), 0.into_process(&mut process));

            assert_eq_in_process!(cons, cons, process);
            assert_eq_in_process!(cons, equal, process);
            assert_ne_in_process!(cons, unequal, process);
        }
    }
}

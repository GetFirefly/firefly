use std::sync::{Arc, RwLock};

use crate::process::identifier::NUMBER_MAX;
use crate::term::Term;

pub fn next() -> Term {
    let counter_arc_rw_lock = ARC_RW_LOCK_COUNTER.clone();
    let mut writable_counter = counter_arc_rw_lock.write().unwrap();

    writable_counter.next()
}

// Private

struct Counter {
    serial: usize,
    number: usize,
}

impl Counter {
    pub fn next(&mut self) -> Term {
        let local_pid = unsafe { Term::local_pid_unchecked(self.number, self.serial) };

        if NUMBER_MAX <= self.number {
            self.serial += 1;
            self.number = 0;
        } else {
            self.number += 1;
        }

        local_pid
    }
}

impl Default for Counter {
    fn default() -> Counter {
        Counter {
            serial: 0,
            number: 0,
        }
    }
}

lazy_static! {
    static ref ARC_RW_LOCK_COUNTER: Arc<RwLock<Counter>> = Default::default();
}

#[cfg(test)]
mod tests {
    use super::*;

    mod counter {
        use super::*;

        mod next {
            use super::*;

            #[test]
            fn number_rolling_over_increments_serial() {
                let mut counter: Counter = Default::default();
                let mut pid = counter.next();

                assert_eq!(pid, Term::local_pid(0, 0).unwrap());

                for _ in 0..NUMBER_MAX + 1 {
                    pid = counter.next();
                }

                assert_eq!(pid, Term::local_pid(0, 1).unwrap());
            }
        }
    }
}

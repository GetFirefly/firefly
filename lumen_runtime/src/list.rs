#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering;

use crate::exception::Exception;
use crate::process::{DebugInProcess, IntoProcess, OrderInProcess, Process, TryIntoInProcess};
use crate::term::{Tag::*, Term};

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

    pub fn to_pid(&self, mut process: &mut Process) -> Result<Term, Exception> {
        let prefix_tail = self.skip_pid_prefix(&mut process)?;
        let prefix_tail_cons: &Cons = prefix_tail.try_into_in_process(&mut process)?;

        let (node, node_tail) = prefix_tail_cons.next_decimal(&mut process)?;
        let node_tail_cons: &Cons = node_tail.try_into_in_process(&mut process)?;

        let first_separator_tail = node_tail_cons.skip_pid_separator(&mut process)?;
        let first_separator_tail_cons: &Cons =
            first_separator_tail.try_into_in_process(&mut process)?;

        let (number, number_tail) = first_separator_tail_cons.next_decimal(&mut process)?;
        let number_tail_cons: &Cons = number_tail.try_into_in_process(&mut process)?;

        let second_separator_tail = number_tail_cons.skip_pid_separator(&mut process)?;
        let second_separator_tail_cons: &Cons =
            second_separator_tail.try_into_in_process(&mut process)?;

        let (serial, serial_tail) = second_separator_tail_cons.next_decimal(&mut process)?;
        let serial_tail_cons: &Cons = serial_tail.try_into_in_process(&mut process)?;

        let suffix_tail = serial_tail_cons.skip_pid_suffix(&mut process)?;

        if suffix_tail.is_empty_list() {
            Term::pid(node, number, serial, &mut process)
        } else {
            Err(bad_argument!(&mut process))
        }
    }

    const PID_PREFIX: Term = unsafe { Term::isize_to_small_integer('<' as isize) };
    const PID_SEPARATOR: Term = unsafe { Term::isize_to_small_integer('.' as isize) };
    const PID_SUFFIX: Term = unsafe { Term::isize_to_small_integer('>' as isize) };

    fn next_decimal(&self, mut process: &mut Process) -> Result<(usize, Term), Exception> {
        self.next_decimal_digit(&mut process)
            .and_then(|(first_digit, first_tail)| {
                Self::rest_decimal_digits(first_digit, first_tail, &mut process)
            })
    }

    fn rest_decimal_digits(
        first_digit: u8,
        first_tail: Term,
        mut process: &mut Process,
    ) -> Result<(usize, Term), Exception> {
        match first_tail.try_into_in_process(&mut process) {
            Ok(first_tail_cons) => {
                let mut acc_decimal: usize = first_digit as usize;
                let mut acc_tail = first_tail;
                let mut acc_cons: &Cons = first_tail_cons;

                while let Ok((digit, tail)) = acc_cons.next_decimal_digit(&mut process) {
                    acc_decimal = 10 * acc_decimal + (digit as usize);
                    acc_tail = tail;

                    match tail.try_into_in_process(&mut process) {
                        Ok(tail_cons) => acc_cons = tail_cons,
                        Err(_) => {
                            break;
                        }
                    }
                }

                Ok((acc_decimal, acc_tail))
            }
            Err(_) => Ok((first_digit as usize, first_tail)),
        }
    }

    fn next_decimal_digit(&self, mut process: &mut Process) -> Result<(u8, Term), Exception> {
        let head = self.head;

        match head.tag() {
            SmallInteger => {
                let head_usize = unsafe { head.small_integer_to_usize() };

                if head_usize <= (std::u8::MAX as usize) {
                    let c: char = head_usize as u8 as char;

                    match c.to_digit(10) {
                        Some(digit) => Ok((digit as u8, self.tail)),
                        None => Err(bad_argument!(&mut process)),
                    }
                } else {
                    Err(bad_argument!(&mut process))
                }
            }
            _ => Err(bad_argument!(&mut process)),
        }
    }

    fn skip_pid_prefix(&self, mut process: &mut Process) -> Result<Term, Exception> {
        if self.head.tagged == Self::PID_PREFIX.tagged {
            Ok(self.tail)
        } else {
            Err(bad_argument!(&mut process))
        }
    }

    fn skip_pid_separator(&self, mut process: &mut Process) -> Result<Term, Exception> {
        if self.head.tagged == Self::PID_SEPARATOR.tagged {
            Ok(self.tail)
        } else {
            Err(bad_argument!(&mut process))
        }
    }

    fn skip_pid_suffix(&self, mut process: &mut Process) -> Result<Term, Exception> {
        if self.head.tagged == Self::PID_SUFFIX.tagged {
            Ok(self.tail)
        } else {
            Err(bad_argument!(&mut process))
        }
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

pub trait ToList {
    fn to_list(&mut self, process: &mut Process) -> Term;
}

impl<T> ToList for T
where
    T: DoubleEndedIterator + Iterator<Item = u8>,
{
    fn to_list(&mut self, mut process: &mut Process) -> Term {
        self.rfold(Term::EMPTY_LIST, |acc, byte| {
            Term::cons(byte.into_process(&mut process), acc, &mut process)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod eq {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};
        use crate::process::IntoProcess;

        #[test]
        fn with_proper() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let cons = Cons::new(0.into_process(&mut process), Term::EMPTY_LIST);
            let equal = Cons::new(0.into_process(&mut process), Term::EMPTY_LIST);
            let unequal = Cons::new(1.into_process(&mut process), Term::EMPTY_LIST);

            assert_eq_in_process!(cons, cons, process);
            assert_eq_in_process!(cons, equal, process);
            assert_ne_in_process!(cons, unequal, process);
        }

        #[test]
        fn with_improper() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let cons = Cons::new(0.into_process(&mut process), 1.into_process(&mut process));
            let equal = Cons::new(0.into_process(&mut process), 1.into_process(&mut process));
            let unequal = Cons::new(1.into_process(&mut process), 0.into_process(&mut process));

            assert_eq_in_process!(cons, cons, process);
            assert_eq_in_process!(cons, equal, process);
            assert_ne_in_process!(cons, unequal, process);
        }
    }
}

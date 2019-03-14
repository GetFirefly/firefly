#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering;
use std::convert::TryInto;

use crate::bad_argument::BadArgument;
use crate::process::{DebugInProcess, IntoProcess, OrderInProcess, Process};
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

    pub fn to_pid(&self, mut process: &mut Process) -> Result<Term, BadArgument> {
        let prefix_tail = self.skip_pid_prefix()?;
        let prefix_tail_cons: &Cons = prefix_tail.try_into()?;

        let (node, node_tail) = prefix_tail_cons.next_decimal()?;
        let node_tail_cons: &Cons = node_tail.try_into()?;

        let first_separator_tail = node_tail_cons.skip_pid_separator()?;
        let first_separator_tail_cons: &Cons = first_separator_tail.try_into()?;

        let (number, number_tail) = first_separator_tail_cons.next_decimal()?;
        let number_tail_cons: &Cons = number_tail.try_into()?;

        let second_separator_tail = number_tail_cons.skip_pid_separator()?;
        let second_separator_tail_cons: &Cons = second_separator_tail.try_into()?;

        let (serial, serial_tail) = second_separator_tail_cons.next_decimal()?;
        let serial_tail_cons: &Cons = serial_tail.try_into()?;

        let suffix_tail = serial_tail_cons.skip_pid_suffix()?;

        if suffix_tail.is_empty_list() {
            Term::pid(node, number, serial, &mut process)
        } else {
            Err(bad_argument!())
        }
    }

    const PID_PREFIX: Term = unsafe { Term::isize_to_small_integer('<' as isize) };
    const PID_SEPARATOR: Term = unsafe { Term::isize_to_small_integer('.' as isize) };
    const PID_SUFFIX: Term = unsafe { Term::isize_to_small_integer('>' as isize) };

    fn next_decimal(&self) -> Result<(usize, Term), BadArgument> {
        self.next_decimal_digit()
            .and_then(&Self::rest_decimal_digits)
    }

    fn rest_decimal_digits(
        (first_digit, first_tail): (u8, Term),
    ) -> Result<(usize, Term), BadArgument> {
        match first_tail.try_into() {
            Ok(first_tail_cons) => {
                let mut acc_decimal: usize = first_digit as usize;
                let mut acc_tail = first_tail;
                let mut acc_cons: &Cons = first_tail_cons;

                while let Ok((digit, tail)) = acc_cons.next_decimal_digit() {
                    acc_decimal = 10 * acc_decimal + (digit as usize);
                    acc_tail = tail;

                    match tail.try_into() {
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

    fn next_decimal_digit(&self) -> Result<(u8, Term), BadArgument> {
        let head = self.head;

        match head.tag() {
            SmallInteger => {
                let head_usize = unsafe { head.small_integer_to_usize() };

                if head_usize <= (std::u8::MAX as usize) {
                    let c: char = head_usize as u8 as char;

                    match c.to_digit(10) {
                        Some(digit) => Ok((digit as u8, self.tail)),
                        None => Err(bad_argument!()),
                    }
                } else {
                    Err(bad_argument!())
                }
            }
            _ => Err(bad_argument!()),
        }
    }

    fn skip_pid_prefix(&self) -> Result<Term, BadArgument> {
        if self.head.tagged == Self::PID_PREFIX.tagged {
            Ok(self.tail)
        } else {
            Err(bad_argument!())
        }
    }

    fn skip_pid_separator(&self) -> Result<Term, BadArgument> {
        if self.head.tagged == Self::PID_SEPARATOR.tagged {
            Ok(self.tail)
        } else {
            Err(bad_argument!())
        }
    }

    fn skip_pid_suffix(&self) -> Result<Term, BadArgument> {
        if self.head.tagged == Self::PID_SUFFIX.tagged {
            Ok(self.tail)
        } else {
            Err(bad_argument!())
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

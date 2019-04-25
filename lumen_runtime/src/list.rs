use std::cmp::Ordering::{self, *};
use std::convert::{TryFrom, TryInto};
#[cfg(test)]
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;

use crate::atom::Existence;
use crate::exception::{self, Exception};
use crate::heap::{CloneIntoHeap, Heap};
use crate::process::{IntoProcess, Process};
use crate::term::{Tag::*, Term};

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

    pub fn clone_into_heap(&self, heap: &Heap) -> Term {
        let mut vec: Vec<Term> = Vec::new();
        let mut initial = Term::EMPTY_LIST;

        for result in self {
            match result {
                Ok(term) => vec.push(term),
                Err(ImproperList { tail }) => initial = tail.clone_into_heap(heap),
            }
        }

        vec.iter().rfold(initial, |acc, term| {
            Term::heap_cons(term.clone_into_heap(heap), acc, &heap)
        })
    }

    pub fn concatenate(&self, term: Term, process: &Process) -> exception::Result {
        match self.into_iter().collect::<Result<Vec<Term>, _>>() {
            Ok(vec) => Ok(Term::vec_to_list(&vec, term, &process)),
            Err(ImproperList { .. }) => Err(badarg!()),
        }
    }

    pub fn contains(&self, term: Term) -> bool {
        self.into_iter().any(|result| match result {
            Ok(ref element) => element == &term,
            _ => false,
        })
    }

    pub fn is_proper(&self) -> bool {
        self.into_iter().all(|item| item.is_ok())
    }

    pub fn is_char_list(&self) -> bool {
        self.into_iter().all(|result| match result {
            Ok(term) => char::try_from(term).is_ok(),
            Err(_) => false,
        })
    }

    pub fn subtract(&self, subtrahend: &Cons, process: &Process) -> exception::Result {
        match self.into_iter().collect::<Result<Vec<Term>, _>>() {
            Ok(mut self_vec) => {
                for result in subtrahend.into_iter() {
                    match result {
                        Ok(subtrahend_element) => self_vec.remove_item(&subtrahend_element),
                        Err(ImproperList { .. }) => return Err(badarg!()),
                    };
                }

                Ok(Term::vec_to_list(&self_vec, Term::EMPTY_LIST, &process))
            }
            Err(ImproperList { .. }) => Err(badarg!()),
        }
    }

    pub fn to_atom(&self, existence: Existence) -> exception::Result {
        let string: String = self
            .into_iter()
            .map(|result| match result {
                Ok(term) => char::try_from(term),
                Err(ImproperList { .. }) => Err(badarg!()),
            })
            .collect::<Result<String, Exception>>()?;

        Term::str_to_atom(&string, existence).ok_or_else(|| badarg!())
    }

    pub fn to_pid(&self, process: &Process) -> exception::Result {
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
            Term::pid(node, number, serial, &process)
        } else {
            Err(badarg!())
        }
    }

    pub fn to_tuple(&self, process: &Process) -> exception::Result {
        let vec: Vec<Term> = self
            .into_iter()
            .collect::<Result<Vec<Term>, _>>()
            .map_err(|_| badarg!())?;

        Ok(Term::slice_to_tuple(vec.as_slice(), &process))
    }

    const PID_PREFIX: Term = unsafe { Term::isize_to_small_integer('<' as isize) };
    const PID_SEPARATOR: Term = unsafe { Term::isize_to_small_integer('.' as isize) };
    const PID_SUFFIX: Term = unsafe { Term::isize_to_small_integer('>' as isize) };

    fn next_decimal(&self) -> Result<(usize, Term), Exception> {
        self.next_decimal_digit()
            .and_then(|(first_digit, first_tail)| {
                Self::rest_decimal_digits(first_digit, first_tail)
            })
    }

    fn rest_decimal_digits(first_digit: u8, first_tail: Term) -> Result<(usize, Term), Exception> {
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

    fn next_decimal_digit(&self) -> Result<(u8, Term), Exception> {
        let head = self.head;

        match head.tag() {
            SmallInteger => {
                let head_usize = unsafe { head.small_integer_to_usize() };

                if head_usize <= (std::u8::MAX as usize) {
                    let c: char = head_usize as u8 as char;

                    match c.to_digit(10) {
                        Some(digit) => Ok((digit as u8, self.tail)),
                        None => Err(badarg!()),
                    }
                } else {
                    Err(badarg!())
                }
            }
            _ => Err(badarg!()),
        }
    }

    fn skip_pid_prefix(&self) -> exception::Result {
        if self.head.tagged == Self::PID_PREFIX.tagged {
            Ok(self.tail)
        } else {
            Err(badarg!())
        }
    }

    fn skip_pid_separator(&self) -> exception::Result {
        if self.head.tagged == Self::PID_SEPARATOR.tagged {
            Ok(self.tail)
        } else {
            Err(badarg!())
        }
    }

    fn skip_pid_suffix(&self) -> exception::Result {
        if self.head.tagged == Self::PID_SUFFIX.tagged {
            Ok(self.tail)
        } else {
            Err(badarg!())
        }
    }
}

#[cfg(test)]
impl Debug for Cons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cons::new({:?}, {:?})", self.head, self.tail)
    }
}

impl Eq for Cons {}

impl Hash for Cons {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.head.hash(state);
        self.tail.hash(state);
    }
}

#[derive(Clone)]
pub struct ImproperList {
    tail: Term,
}

impl IntoIterator for &Cons {
    type Item = Result<Term, ImproperList>;
    type IntoIter = Iter;

    fn into_iter(self) -> Iter {
        Iter {
            head: Some(Ok(self.head)),
            tail: Some(self.tail),
        }
    }
}

pub struct Iter {
    head: Option<Result<Term, ImproperList>>,
    tail: Option<Term>,
}

impl FusedIterator for Iter {}

impl Iterator for Iter {
    type Item = Result<Term, ImproperList>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.head.clone();

        match next {
            None => (),
            Some(Err(_)) => {
                self.head = None;
                self.tail = None;
            }
            _ => {
                let tail = self.tail.unwrap();

                match tail.tag() {
                    EmptyList => {
                        self.head = None;
                        self.tail = None;
                    }
                    List => {
                        let cons: &Cons = unsafe { tail.as_ref_cons_unchecked() };

                        self.head = Some(Ok(cons.head));
                        self.tail = Some(cons.tail);
                    }
                    _ => {
                        self.head = Some(Err(ImproperList { tail }));
                        self.tail = None;
                    }
                }
            }
        }

        next
    }
}

impl PartialOrd for Cons {
    fn partial_cmp(&self, other: &Cons) -> Option<Ordering> {
        match self.head.partial_cmp(&other.head) {
            Some(Equal) => self.tail.partial_cmp(&other.tail),
            partial_ordering => partial_ordering,
        }
    }
}

impl PartialEq for Cons {
    fn eq(&self, other: &Cons) -> bool {
        self.head == other.head && self.tail == other.tail
    }
}

pub trait ToList {
    fn to_list(&mut self, process: &Process) -> Term;
}

impl<T> ToList for T
where
    T: DoubleEndedIterator + Iterator<Item = u8>,
{
    fn to_list(&mut self, process: &Process) -> Term {
        self.rfold(Term::EMPTY_LIST, |acc, byte| {
            Term::cons(byte.into_process(&process), acc, &process)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::process;

    mod eq {
        use super::*;

        use crate::process::IntoProcess;

        #[test]
        fn with_proper() {
            let process = process::local::new();
            let cons = Cons::new(0.into_process(&process), Term::EMPTY_LIST);
            let equal = Cons::new(0.into_process(&process), Term::EMPTY_LIST);
            let unequal = Cons::new(1.into_process(&process), Term::EMPTY_LIST);

            assert_eq!(cons, cons);
            assert_eq!(cons, equal);
            assert_ne!(cons, unequal);
        }

        #[test]
        fn with_improper() {
            let process = process::local::new();
            let cons = Cons::new(0.into_process(&process), 1.into_process(&process));
            let equal = Cons::new(0.into_process(&process), 1.into_process(&process));
            let unequal = Cons::new(1.into_process(&process), 0.into_process(&process));

            assert_eq!(cons, cons);
            assert_eq!(cons, equal);
            assert_ne!(cons, unequal);
        }
    }
}

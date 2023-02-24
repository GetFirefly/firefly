use core::marker::PhantomData;

use crate::term::{OpaqueTerm, Term};

use super::*;

pub struct Iter<'a> {
    head: Option<Result<Term, ImproperList>>,
    tail: Option<OpaqueTerm>,
    _marker: PhantomData<&'a Cons>,
}
impl Iter<'_> {
    pub(super) fn new(cons: &Cons) -> Self {
        Self {
            head: Some(Ok(cons.head())),
            tail: Some(cons.tail),
            _marker: PhantomData,
        }
    }
}

impl core::iter::FusedIterator for Iter<'_> {}

impl Iterator for Iter<'_> {
    type Item = Result<Term, ImproperList>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.head.take();

        match next {
            None => next,
            Some(Err(_)) => {
                self.head = None;
                self.tail = None;
                next
            }
            Some(Ok(Term::Nil)) if self.tail.is_none() => {
                self.head = None;
                self.tail = None;
                None
            }
            next => {
                let tail = self.tail.unwrap();
                match tail.into() {
                    Term::Nil => {
                        self.head = Some(Ok(Term::Nil));
                        self.tail = None;
                        next
                    }
                    Term::Cons(cons) => {
                        self.head = Some(Ok(cons.head()));
                        self.tail = Some(cons.tail);
                        next
                    }
                    Term::None => panic!("invalid none value found in list"),
                    tail => {
                        self.head = Some(Err(ImproperList { tail }));
                        self.tail = None;
                        next
                    }
                }
            }
        }
    }
}

pub struct IterRaw<'a> {
    head: Option<Result<OpaqueTerm, OpaqueTerm>>,
    tail: Option<OpaqueTerm>,
    _marker: PhantomData<&'a Cons>,
}
impl IterRaw<'_> {
    pub(super) fn new(cons: &Cons) -> Self {
        Self {
            head: Some(Ok(cons.head)),
            tail: Some(cons.tail),
            _marker: PhantomData,
        }
    }
}

impl core::iter::FusedIterator for IterRaw<'_> {}

impl Iterator for IterRaw<'_> {
    type Item = Result<OpaqueTerm, OpaqueTerm>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.head.take();

        match next {
            None => next,
            Some(Err(_)) => {
                self.head = None;
                self.tail = None;
                next
            }
            Some(Ok(OpaqueTerm::NIL)) if self.tail.is_none() => {
                self.head = None;
                self.tail = None;
                None
            }
            next => {
                let tail = self.tail.unwrap();
                assert_ne!(tail, OpaqueTerm::NONE);
                if tail.is_nil() {
                    self.head = Some(Ok(tail));
                    self.tail = None;
                    return next;
                }
                if tail.is_nonempty_list() {
                    let cons = unsafe { &*(tail.as_ptr() as *const Cons) };
                    self.head = Some(Ok(cons.head));
                    self.tail = Some(cons.tail);
                    return next;
                }
                self.head = Some(Err(tail));
                self.tail = None;
                next
            }
        }
    }
}

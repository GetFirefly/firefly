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

pub struct CellsIter<'a> {
    current: Option<&'a mut Cons>,
}
impl<'a> CellsIter<'a> {
    pub(super) fn new(cons: &'a mut Cons) -> Self {
        Self {
            current: Some(cons),
        }
    }
}
impl core::iter::FusedIterator for CellsIter<'_> {}

impl<'a> Iterator for CellsIter<'a> {
    type Item = &'a mut Cons;

    fn next(&mut self) -> Option<Self::Item> {
        let cell = self.current.take()?;

        if cell.tail.is_nonempty_list() {
            self.current = Some(unsafe { &mut *(cell.tail.as_ptr() as *mut Cons) });
        }

        Some(cell)
    }
}

pub struct RawIter<'a> {
    head: Option<Result<OpaqueTerm, OpaqueTerm>>,
    tail: Option<OpaqueTerm>,
    _marker: PhantomData<&'a Cons>,
}
impl RawIter<'_> {
    pub(super) fn new(cons: &Cons) -> Self {
        Self {
            head: Some(Ok(cons.head)),
            tail: Some(cons.tail),
            _marker: PhantomData,
        }
    }
}

impl core::iter::FusedIterator for RawIter<'_> {}

impl Iterator for RawIter<'_> {
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

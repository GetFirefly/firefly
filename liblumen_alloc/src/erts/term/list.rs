use core::cmp;
use core::iter::FusedIterator;

use super::{AsTerm, Term, TypedTerm};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaybeImproper<P, I> {
    Proper(P),
    Improper(I),
}
impl<P, I> MaybeImproper<P, I> {
    #[inline]
    pub fn is_proper(&self) -> bool {
        match self {
            &Self::Proper(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_improper(&self) -> bool {
        !self.is_proper()
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Cons {
    pub head: Term,
    pub tail: Term,
}
impl Cons {
    /// Create a new cons cell from a head and tail pointer pair
    #[inline]
    pub fn new(head: Term, tail: Term) -> Self {
        Self { head, tail }
    }

    /// Returns true if this cons cell is actually a move marker
    #[inline]
    pub fn is_move_marker(&self) -> bool {
        self.head.is_none()
    }

    /// Reify a cons cell from a pointer to the head of a cons cell
    ///
    /// # Safety
    ///
    /// It is expected that `cons` is a pointer to the `head` of a
    /// previously allocated `Cons`, any other usage may result in
    /// undefined behavior or a segmentation fault
    #[inline]
    pub unsafe fn from_raw(cons: *mut Cons) -> Self {
        *cons
    }

    /// Get the `TypedTerm` pointed to by the head of this cons cell
    #[inline]
    pub fn head(&self) -> TypedTerm {
        unsafe { self.head.to_typed_term().unwrap() }
    }

    /// Get the tail of this cons cell, which depending on the type of
    /// list it represents, may either be another `Cons`, a `TypedTerm`
    /// value, or no value at all (when the tail is nil).
    ///
    /// If the list is improper, then `Some(TypedTerm)` will be returned.
    /// If the list is proper, then either `Some(TypedTerm)` or `Nil` will
    /// be returned, depending on whether this cell is the last in the list.
    #[inline]
    pub fn tail(&self) -> Option<MaybeImproper<Cons, TypedTerm>> {
        match unsafe { self.tail.to_typed_term() } {
            None => None,
            Some(TypedTerm::Nil) => None,
            Some(TypedTerm::List(tail)) => Some(MaybeImproper::Proper(*tail)),
            Some(other) => Some(MaybeImproper::Improper(other)),
        }
    }

    /// Constructs an iterator for the list represented by this cons cell
    #[inline]
    pub fn iter(&self) -> ListIter {
        ListIter::new(*self)
    }
}
unsafe impl AsTerm for Cons {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::from_raw((&self.head as *const _ as usize) | Term::FLAG_LIST)
    }
}
impl PartialEq<Cons> for Cons {
    fn eq(&self, other: &Cons) -> bool {
        self.head.eq(&other.head) && self.tail.eq(&other.tail)
    }
}
impl PartialOrd<Cons> for Cons {
    fn partial_cmp(&self, other: &Cons) -> Option<cmp::Ordering> {
        self.iter()
            .map(|t| unsafe { t.to_typed_term().unwrap() })
            .partial_cmp(other.iter().map(|t| unsafe { t.to_typed_term().unwrap() }))
    }
}

pub struct ListIter {
    head: Option<Cons>,
    tail: Option<MaybeImproper<Cons, TypedTerm>>,
    pos: usize,
    panic_on_improper: bool,
}
impl ListIter {
    /// Creates a new list iterator which works for both improper and proper lists
    #[inline]
    pub fn new(cons: Cons) -> Self {
        let pos = 0;
        let panic_on_improper = false;
        Self {
            head: Some(cons),
            tail: None,
            pos,
            panic_on_improper,
        }
    }

    /// Creates a new list itertator which panics if the list is improper
    #[inline]
    pub fn new_strict(cons: Cons) -> Self {
        let pos = 0;
        let panic_on_improper = true;
        Self {
            head: Some(cons),
            tail: None,
            pos,
            panic_on_improper,
        }
    }
}
impl Iterator for ListIter {
    type Item = Term;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.head {
                Some(cons) => {
                    if cons.head.is_nil() {
                        return None;
                    }
                    self.pos += 1;
                    self.head = None;
                    self.tail = cons.tail();
                    return Some(cons.head);
                }
                None => match self.tail {
                    Some(MaybeImproper::Improper(_)) if self.panic_on_improper => {
                        panic!("tried to iterate over improper list!");
                    }
                    Some(MaybeImproper::Improper(ref value)) => {
                        let val = unsafe { value.as_term() };
                        self.pos += 1;
                        self.head = None;
                        self.tail = None;
                        return Some(val);
                    }
                    Some(MaybeImproper::Proper(cons)) => {
                        self.head = Some(cons);
                        self.tail = None;
                        continue;
                    }
                    None => panic!("called next on an improper list while at the end of the list"),
                },
            }
        }
    }
}
impl<'a> FusedIterator for ListIter {}

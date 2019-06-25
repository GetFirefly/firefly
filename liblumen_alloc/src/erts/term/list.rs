use core::cmp;
use core::ptr;
use core::iter::FusedIterator;
use core::alloc::{AllocErr, Layout};

use crate::borrow::CloneToProcess;
use crate::erts::{AllocInProcess, StackPrimitives};

use super::{AsTerm, Term};

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
    pub fn head(&self) -> Term {
        self.head
    }

    /// Get the tail of this cons cell, which depending on the type of
    /// list it represents, may either be another `Cons`, a `TypedTerm`
    /// value, or no value at all (when the tail is nil).
    ///
    /// If the list is improper, then `Some(TypedTerm)` will be returned.
    /// If the list is proper, then either `Some(TypedTerm)` or `Nil` will
    /// be returned, depending on whether this cell is the last in the list.
    #[inline]
    pub fn tail(&self) -> Option<MaybeImproper<Cons, Term>> {
        if self.tail.is_nil() {
            return None;
        }
        if self.tail.is_list() {
            let tail = self.tail.list_val();
            return Some(MaybeImproper::Proper(unsafe { *tail }));
        }
        Some(MaybeImproper::Improper(self.tail))
    }

    /// Constructs an iterator for the list represented by this cons cell
    /// 
    /// This iterator will panic if it reaches a cell that indicates this
    /// list is an improper list, to safely iterate such lists, use `iter_safe`
    #[inline]
    pub fn iter(&self) -> ListIter {
        ListIter::new_strict(*self)
    }

    /// Same as `iter`, except it will not panic on improper lists, it will
    /// simply return `None` as if it had reached the end of a proper list
    #[inline]
    pub fn iter_safe(&self) -> ListIter {
        ListIter::new(*self)
    }
}
unsafe impl AsTerm for Cons {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_list(self as *const Self)
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
            .map(|t| t.to_typed_term().unwrap())
            .partial_cmp(other.iter().map(|t| t.to_typed_term().unwrap()))
    }
}
impl CloneToProcess for Cons {
    fn clone_to_process<A: AllocInProcess>(&self, process: &mut A) -> Term {
        // To make sure we don't blow the stack, we do not recursively walk
        // the list. Instead we use a list builder and traverse the list iteratively,
        // cloning each element as we go via the builder
        let mut builder = ListBuilder::on_heap(process);
        // Start with the current cell
        let mut current = *self;
        loop {
            // Determine whether we're done, or have more cells to traverse
            if current.tail.is_nil() {
                // End of proper list
                builder = builder.push(current.head);
                return builder.finish().unwrap();
            } else if current.tail.is_list() {
                // Add current element and traverse to the next cell
                current = unsafe { *current.tail.list_val() };
                builder = builder.push(current.head);
                continue;
            } else if current.tail.is_immediate() {
                // End of improper list
                builder = builder.push(current.head);
                return builder.finish_with(current.tail).unwrap();
            } else {
                // End of improper list
                builder = builder.push(current.head);
                return builder.finish_with(current.tail).unwrap();
            }
        }
    }
}

pub struct ListIter {
    head: Option<Term>,
    tail: Option<MaybeImproper<Cons, Term>>,
    pos: usize,
    panic_on_improper: bool,
}
impl ListIter {
    /// Creates a new list iterator which works for both improper and proper lists
    #[inline]
    pub fn new(cons: Cons) -> Self {
        assert!(!cons.is_move_marker());
        let pos = 0;
        let panic_on_improper = false;
        Self {
            head: Some(cons.head),
            tail: cons.tail(),
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
            head: Some(cons.head),
            tail: cons.tail(),
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
                Some(head) => {
                    assert!(!head.is_none(), "unexpected move marker in cons cell");
                    self.pos += 1;
                    self.head = None;
                    return Some(head);
                }
                None => match self.tail {
                    Some(MaybeImproper::Improper(_)) if self.panic_on_improper => {
                        panic!("tried to iterate over improper list!");
                    }
                    Some(MaybeImproper::Improper(tail)) => {
                        self.pos += 1;
                        self.head = None;
                        self.tail = None;
                        return Some(tail);
                    }
                    Some(MaybeImproper::Proper(cons)) => {
                        self.head = Some(cons.head);
                        self.tail = cons.tail();
                        continue;
                    }
                    None => {
                        return None;
                    }
                },
            }
        }
    }
}
impl<'a> FusedIterator for ListIter {}

/// A trait for the way allocation of cons cells is performed by `ListBuilder`
pub trait ListAllocationType {}

use heapless::Vec;
#[cfg(target_pointer_width = "64")]
use heapless::consts::U8;
#[cfg(target_pointer_width = "32")]
use heapless::consts::U16;

/// AllocationType that indicates cons cells will be allocated on the stack
/// 
/// Lists allocated on the stack are allowed 64 bytes, or 8 elements on a 64-bit
/// target, 16 elements on a 32-bit target. If the allocation exceeds that amount
/// an AllocErr will be returned
#[cfg(target_pointer_width = "64")]
pub struct OnStack(Vec<Term, U8>);
#[cfg(target_pointer_width = "32")]
pub struct OnStack(Vec<Term, U16>);
impl ListAllocationType for OnStack {}

/// AllocationType that indicates cons cells will be allocated on the heap
pub struct OnHeap;
impl ListAllocationType for OnHeap {}

pub struct ListBuilder<'a, T: ListAllocationType, A: AllocInProcess> {
    process: &'a mut A,
    element: Option<Term>,
    first: *mut Cons,
    current: *mut Cons,
    size: usize,
    mode: T,
    failed: bool,
}
impl<'a, A: AllocInProcess> ListBuilder<'a, OnHeap, A> {
    /// Creates a new list builder that allocates on the given processes' heap
    #[inline]
    pub fn on_heap(process: &'a mut A) -> Self {
        Self {
            process,
            element: None,
            first: ptr::null_mut(),
            current: ptr::null_mut(),
            size: 0,
            mode: OnHeap,
            failed: false,
        }
    }

    /// Pushes the given `Term` on to the end of the list being built.
    /// 
    /// NOTE: When using `on_heap` mode, the builder will also clone terms if they are
    /// not already on the process heap
    #[inline]
    pub fn push(mut self, term: Term) -> Self {
        if self.failed {
            self
        } else {
            match self.push_internal(term) {
                Err(_) => {
                    self.failed = true;
                    self
                }
                Ok(_) => {
                    self
                }
            }
        }
    }

    #[inline]
    fn push_internal(&mut self, term: Term) -> Result<(), AllocErr> {
        if self.element.is_none() {
            // This is the very first push
            assert!(self.first.is_null());
            // We need to allocate a cell
            self.first = self.alloc_cell(Term::NIL, Term::NIL)?;
            self.element = Some(self.clone_term_to_process(term));
            self.current = self.first;
            self.size += 1;
        } else {
            let new_element = self.clone_term_to_process(term);
            // Swap the current element with the new element
            let head = self.element.replace(new_element).unwrap();
            // Set the head of the current cell to the element we just took out
            let mut current = unsafe { &mut *self.current };
            if current.head.is_nil() {
                // Fill the head of the cell, this only happens on the second push
                current.head = head;
                self.size += 1;
            } else if current.tail.is_nil() {
                // Fill the tail of the cell, this is the typical case
                current.tail = head;
                self.size += 1;
            } else {
                // This occurs when a cell has been filled and we're pushing a new element.
                // We need to allocate a new cell and move the previous tail to the head of that cell
                let cadr = current.tail;
                let new_current = self.alloc_cell(cadr, Term::NIL)?;
                current.tail = Term::make_list(new_current);
                self.current = new_current;
                self.size += 1;
            }
        }
        Ok(())
    }

    /// Consumes the builder and produces a `Term` which points to the allocated list
    #[inline]
    pub fn finish(mut self) -> Result<Term, AllocErr> {
        if self.failed {
            // We can't clean up the heap until GC
            Err(AllocErr)
        } else {
            match self.element {
                // Empty list
                None => {
                    assert!(self.first.is_null());
                    Ok(Term::NIL)
                }
                // Non-empty list
                Some(last) => {
                    let mut current = unsafe { &mut *self.current };
                    if current.head.is_nil() {
                        // Occurs with one element lists, as tail cell allocation is deferred
                        current.head = last;
                    } else if current.tail.is_nil() {
                        // Typical case, we need to allocate a new tail to wrap up this list
                        let new_current = self.alloc_cell(last, Term::NIL)?;
                        current.tail = Term::make_list(new_current);
                    } else {
                        // This occurs when we've filled a cell, so we need to allocate two cells,
                        // one for the current tail and one for the last element
                        let cadr_cell = self.alloc_cell(last, Term::NIL)?;
                        let cadr = Term::make_list(cadr_cell);
                        let cdr = self.alloc_cell(current.tail, cadr)?;
                        current.tail = Term::make_list(cdr);
                    }
                    // Return a reference to the first cell of the list
                    Ok(Term::make_list(self.first))
                }
            }
        }
    }

    /// Like `finish`, but produces an improper list if there is already at least one element
    #[inline]
    pub fn finish_with(mut self, term: Term) -> Result<Term, AllocErr> {
        if self.failed {
            // We can't clean up the heap until GC
            Err(AllocErr)
        } else {
            match self.element {
                // Empty list
                None => {
                    // We need to allocate a cell, this will end up a proper list
                    self.push_internal(term)?;
                    self.finish()
                }
                // Non-empty list, this will always produce an improper list
                Some(next) => {
                    let mut current = unsafe { &mut *self.current };
                    if current.head.is_nil() {
                        current.head = next;
                        current.tail = self.clone_term_to_process(term);
                    } else if current.tail.is_nil() {
                        let tail = self.clone_term_to_process(term);
                        let cdr = self.alloc_cell(next, tail)?;
                        current.tail = Term::make_list(cdr);
                    } else {
                        let tail = self.clone_term_to_process(term);
                        let cadr_cell = self.alloc_cell(next, tail)?;
                        let cadr = Term::make_list(cadr_cell);
                        let cdr = self.alloc_cell(current.tail, cadr)?;
                        current.tail = Term::make_list(cdr);
                    }
                    Ok(Term::make_list(self.first))
                }
            }
        }
    }

    #[inline]
    fn clone_term_to_process(&mut self, term: Term) -> Term {
        if term.is_immediate() {
            term
        } else if term.is_boxed() {
            let ptr = term.boxed_val();
            if !term.is_literal() && self.process.is_owner(ptr) {
                // No need to clone
                term
            } else {
                term.clone_to_process(self.process)
            }
        } else if term.is_list() {
            let ptr = term.list_val();
            if !term.is_literal() && self.process.is_owner(ptr) {
                // No need to clone
                term
            } else {
                term.clone_to_process(self.process)
            }
        } else {
            unreachable!()
        }
    }

    #[inline]
    fn alloc_cell(&mut self, head: Term, tail: Term) -> Result<*mut Cons, AllocErr> {
        let ptr = unsafe { self.process.alloc_layout(Layout::new::<Cons>())?.as_ptr() as *mut Cons };
        let mut cell = unsafe { &mut *ptr };
        cell.head = head;
        cell.tail = tail;
        Ok(ptr)
    }
}
impl<'a, A: AllocInProcess + StackPrimitives> ListBuilder<'a, OnStack, A> {
    /// Creates a new list builder that allocates on the given processes' heap
    #[inline]
    pub fn on_stack(process: &'a mut A) -> Self {
        Self {
            process,
            element: None,
            first: ptr::null_mut(),
            current: ptr::null_mut(),
            size: 0,
            mode: OnStack(Vec::new()),
            failed: false,
        }
    } 

    /// Pushes the given `Term` on to the end of the list being built.
    /// 
    /// NOTE: It is not permitted to allocate on the stack while constructing
    /// a stack-allocated list, furthermore, stack-allocated lists may only consist
    /// of immediates or boxes. If these invariants are violated, this function will panic.
    #[inline]
    pub fn push(mut self, term: Term) -> Self {
        assert!(term.is_immediate() || term.is_boxed() || term.is_list(), "invalid list element for stack-allocated list");
        if self.failed {
            self
        } else {
            match self.mode.0.push(term) {
                Err(_) => {
                    self.failed = true;
                    self
                }
                Ok(_) => self,
            }
        }
    }

    /// Consumes the builder and produces a `Term` which points to the allocated list
    #[inline]
    pub fn finish(self) -> Result<Term, AllocErr> {
        if self.failed {
            Err(AllocErr)
        } else {
            let size = self.mode.0.len();
            if size == 0 {
                unsafe {
                    let ptr = self.process.alloca(1)?.as_ptr();
                    ptr::write(ptr, Term::NIL);
                }
                Ok(Term::NIL)
            } else {
                // Construct allocation layout for list
                let (elements_layout, _) = Layout::new::<Cons>()
                    .repeat(size)
                    .unwrap();
                let (layout, _) = Layout::new::<Term>()
                    .extend(elements_layout)
                    .unwrap();
                // Allocate on stack
                let ptr = unsafe { self.process.alloca_layout(layout)?.as_ptr() };
                // Get pointer to first cell
                let first_ptr = unsafe { ptr.offset(1) as *mut Cons };
                // Write header with pointer to first cell
                unsafe { ptr::write(ptr, Term::make_list(first_ptr)) };
                // For each element in the list, write a cell with a pointer to the next one
                for (i, element) in self.mode.0.iter().copied().enumerate() {
                    // Offsets are relative to the first cell, first element has `i` of 0
                    let cell_ptr = unsafe { first_ptr.offset(i as isize) };
                    // Get mutable reference to cell memory
                    let mut cell = unsafe { &mut *cell_ptr };
                    if i < size - 1 {
                        // If we have future cells to write, generate a valid tail
                        let tail_ptr = unsafe { cell_ptr.offset(1) };
                        cell.head = element;
                        cell.tail = Term::make_list(tail_ptr);
                    } else {
                        // This is the last element
                        cell.head = element;
                        cell.tail = Term::NIL;
                    }
                }
                Ok(unsafe { *ptr })
            }
        }
    }

    /// Like `finish`, but produces an improper list if there is already at least one element
    #[inline]
    pub fn finish_with(mut self, term: Term) -> Result<Term, AllocErr> {
        if self.failed {
            Err(AllocErr)
        } else {
            let size = self.mode.0.len() + 1;
            if size == 1 {
                // Proper list
                self.mode.0.push(term).unwrap();
                self.finish()
            } else {
                // Construct allocation layout for list
                let (elements_layout, _) = Layout::new::<Cons>()
                    .repeat(size)
                    .unwrap();
                let (layout, _) = Layout::new::<Term>()
                    .extend(elements_layout)
                    .unwrap();
                // Allocate on stack
                let ptr = unsafe { self.process.alloca_layout(layout)?.as_ptr() };
                // Get pointer to first cell
                let first_ptr = unsafe { ptr.offset(1) as *mut Cons };
                // Write header with pointer to first cell
                unsafe { ptr::write(ptr, Term::make_list(first_ptr)) };
                // For each element in the list, write a cell with a pointer to the next one
                for (i, element) in self.mode.0.iter().copied().enumerate() {
                    // Offsets are relative to the first cell, first element has `i` of 0
                    let cell_ptr = unsafe { first_ptr.offset(i as isize) };
                    // Get mutable reference to cell memory
                    let mut cell = unsafe { &mut *cell_ptr };
                    if i < size - 2 {
                        // If we have future cells to write, generate a valid tail
                        let tail_ptr = unsafe { cell_ptr.offset(1) };
                        cell.head = element;
                        cell.tail = Term::make_list(tail_ptr);
                    } else {
                        // This is the last cell
                        cell.head = element;
                        cell.tail = term;
                    }
                }
                Ok(unsafe { *ptr })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erts::SmallInteger;

    #[test]
    fn proper_list_iter_test() {
        let a = unsafe { SmallInteger::new(42).unwrap().as_term() };
        let b = unsafe { SmallInteger::new(24).unwrap().as_term() };
        let c = unsafe { SmallInteger::new(11).unwrap().as_term() };
        let last = Cons::new(c, Term::NIL);
        let last_term = Term::make_list(&last);
        let second = Cons::new(b, last_term);
        let second_term = Term::make_list(&second);
        let first = Cons::new(a, second_term);

        let mut list_iter = first.iter_safe();
        let l0 = list_iter.next().unwrap();
        assert_eq!(l0, a);
        let l1 = list_iter.next().unwrap();
        assert_eq!(l1, b);
        let l2 = list_iter.next().unwrap();
        assert_eq!(l2, c);
        assert_eq!(list_iter.next(), None);
        assert_eq!(list_iter.next(), None);
    }

    #[test]
    fn improper_list_iter_safe_test() {
        let a = unsafe { SmallInteger::new(42).unwrap().as_term() };
        let b = unsafe { SmallInteger::new(24).unwrap().as_term() };
        let c = unsafe { SmallInteger::new(11).unwrap().as_term() };
        let d = unsafe { SmallInteger::new(99).unwrap().as_term() };
        let last = Cons::new(c, d);
        let last_term = Term::make_list(&last);
        let second = Cons::new(b, last_term);
        let second_term = Term::make_list(&second);
        let first = Cons::new(a, second_term);

        let mut list_iter = first.iter_safe();
        let l0 = list_iter.next().unwrap();
        assert_eq!(l0, a);
        let l1 = list_iter.next().unwrap();
        assert_eq!(l1, b);
        let l2 = list_iter.next().unwrap();
        assert_eq!(l2, c);
        let l3 = list_iter.next().unwrap();
        assert_eq!(l3, d);
        assert_eq!(list_iter.next(), None);
        assert_eq!(list_iter.next(), None);
    }

    #[test]
    #[should_panic(expected = "tried to iterate over improper list!")]
    fn improper_list_strict_iter_test() {
        let a = unsafe { SmallInteger::new(42).unwrap().as_term() };
        let b = unsafe { SmallInteger::new(24).unwrap().as_term() };
        let c = unsafe { SmallInteger::new(11).unwrap().as_term() };
        let d = unsafe { SmallInteger::new(99).unwrap().as_term() };
        let last = Cons::new(c, d);
        let last_term = Term::make_list(&last);
        let second = Cons::new(b, last_term);
        let second_term = Term::make_list(&second);
        let first = Cons::new(a, second_term);

        let mut list_iter = first.iter();
        let l0 = list_iter.next().unwrap();
        assert_eq!(l0, a);
        let l1 = list_iter.next().unwrap();
        assert_eq!(l1, b);
        let l2 = list_iter.next().unwrap();
        assert_eq!(l2, c);
        let l3 = list_iter.next().unwrap();
        assert_eq!(l3, d);
        assert_eq!(list_iter.next(), None);
    }
}
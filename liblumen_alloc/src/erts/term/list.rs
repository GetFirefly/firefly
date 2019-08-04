use core::alloc::Layout;
use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug};
use core::iter::FusedIterator;
use core::mem;
use core::ptr;

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::term::{AsTerm, Boxed, Term, TypeError, TypedTerm};
use crate::erts::{to_word_size, HeapAlloc, StackAlloc};

pub enum List {
    Empty,
    NonEmpty(Boxed<Cons>),
}

impl TryFrom<Term> for List {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<List, TypeError> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for List {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<List, TypeError> {
        match typed_term {
            TypedTerm::Nil => Ok(List::Empty),
            TypedTerm::List(cons) => Ok(List::NonEmpty(cons)),
            _ => Err(TypeError),
        }
    }
}

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

#[derive(Clone, Copy, Hash)]
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

    pub fn contains(&self, term: Term) -> bool {
        self.into_iter().any(|result| match result {
            Ok(ref element) => element == &term,
            _ => false,
        })
    }

    /// Returns the count only if the list is proper.
    pub fn count(&self) -> Option<usize> {
        let mut count = 0;

        for result in self {
            match result {
                Ok(_) => count += 1,
                Err(_) => return None,
            }
        }

        Some(count)
    }

    /// Returns true if this cons cell is actually a move marker
    #[inline]
    pub fn is_move_marker(&self) -> bool {
        self.head.is_none()
    }

    /// Returns true if this cons is the head oflumen_runtime/src/otp/erlang.rs:1818:26
    ///a proper list.
    pub fn is_proper(&self) -> bool {
        self.into_iter().all(|result| result.is_ok())
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
        if self.tail.is_non_empty_list() {
            let tail = self.tail.list_val();
            return Some(MaybeImproper::Proper(unsafe { *tail }));
        }
        Some(MaybeImproper::Improper(self.tail))
    }
}
unsafe impl AsTerm for Cons {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_list(self as *const Self)
    }
}
impl Debug for Cons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:?} | {:?}]", self.head, self.tail)
    }
}
impl PartialEq<Cons> for Cons {
    fn eq(&self, other: &Cons) -> bool {
        self.head.eq(&other.head) && self.tail.eq(&other.tail)
    }
}
impl PartialOrd<Cons> for Cons {
    fn partial_cmp(&self, other: &Cons) -> Option<cmp::Ordering> {
        self.into_iter().partial_cmp(other.into_iter())
    }
}
impl CloneToProcess for Cons {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        // To make sure we don't blow the stack, we do not recursively walk
        // the list. Instead we use a list builder and traverse the list iteratively,
        // cloning each element as we go via the builder
        let mut builder = ListBuilder::new(heap);
        // Start with the current cell
        let mut current = self;
        loop {
            builder = builder.push(current.head);

            // Determine whether we're done, or have more cells to traverse
            if current.tail.is_non_empty_list() {
                // Traverse to the next cell
                current = unsafe { &*current.tail.list_val() };
                continue;
            } else if current.tail.is_nil() {
                // End of proper list
                return builder.finish();
            } else {
                // End of improper list
                return builder.finish_with(current.tail);
            }
        }
    }

    fn size_in_words(&self) -> usize {
        let mut elements = 0;
        let mut words = 0;

        for result in self.into_iter() {
            let element = match result {
                Ok(element) => element,
                Err(ImproperList { tail }) => tail,
            };

            elements += 1;
            words += element.size_in_words();
        }

        words += to_word_size(elements * mem::size_of::<Cons>());
        words
    }
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

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd)]
pub struct ImproperList {
    pub tail: Term,
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

                match tail.to_typed_term().unwrap() {
                    TypedTerm::Nil => {
                        self.head = None;
                        self.tail = None;
                    }
                    TypedTerm::List(cons) => {
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

impl TryFrom<Term> for Boxed<Cons> {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Boxed<Cons> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::List(cons) => Ok(cons),
            _ => Err(TypeError),
        }
    }
}

pub struct ListBuilder<'a, A: HeapAlloc> {
    heap: &'a mut A,
    element: Option<Term>,
    first: *mut Cons,
    current: *mut Cons,
    size: usize,
    failed: bool,
}
impl<'a, A: HeapAlloc> ListBuilder<'a, A> {
    /// Creates a new list builder that allocates on the given processes' heap
    #[inline]
    pub fn new(heap: &'a mut A) -> Self {
        Self {
            heap,
            element: None,
            first: ptr::null_mut(),
            current: ptr::null_mut(),
            size: 0,
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
                Ok(_) => self,
            }
        }
    }

    #[inline]
    fn push_internal(&mut self, term: Term) -> Result<(), Alloc> {
        if self.element.is_none() {
            // This is the very first push
            assert!(self.first.is_null());
            // We need to allocate a cell
            self.first = self.alloc_cell(Term::NIL, Term::NIL)?;
            self.element = Some(self.clone_term_to_heap(term)?);
            self.current = self.first;
            self.size += 1;
        } else {
            let new_element = self.clone_term_to_heap(term)?;
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
                // We need to allocate a new cell and move the previous tail to the head of that
                // cell
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
    pub fn finish(mut self) -> Result<Term, Alloc> {
        if self.failed {
            // We can't clean up the heap until GC
            Err(alloc!())
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
    pub fn finish_with(mut self, term: Term) -> Result<Term, Alloc> {
        if self.failed {
            // We can't clean up the heap until GC
            Err(alloc!())
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
                        current.tail = self.clone_term_to_heap(term)?;
                    } else if current.tail.is_nil() {
                        let tail = self.clone_term_to_heap(term)?;
                        let cdr = self.alloc_cell(next, tail)?;
                        current.tail = Term::make_list(cdr);
                    } else {
                        let tail = self.clone_term_to_heap(term)?;
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
    fn clone_term_to_heap(&mut self, term: Term) -> Result<Term, Alloc> {
        if term.is_immediate() {
            Ok(term)
        } else if term.is_boxed() {
            let ptr = term.boxed_val();
            if !term.is_literal() && self.heap.is_owner(ptr) {
                // No need to clone
                Ok(term)
            } else {
                term.clone_to_heap(self.heap)
            }
        } else if term.is_non_empty_list() {
            let ptr = term.list_val();
            if !term.is_literal() && self.heap.is_owner(ptr) {
                // No need to clone
                Ok(term)
            } else {
                term.clone_to_heap(self.heap)
            }
        } else {
            unreachable!()
        }
    }

    #[inline]
    fn alloc_cell(&mut self, head: Term, tail: Term) -> Result<*mut Cons, Alloc> {
        let ptr = unsafe { self.heap.alloc_layout(Layout::new::<Cons>())?.as_ptr() as *mut Cons };
        let mut cell = unsafe { &mut *ptr };
        cell.head = head;
        cell.tail = tail;
        Ok(ptr)
    }
}

#[allow(non_camel_case_types)]
#[cfg(target_pointer_width = "32")]
type MAX_ELEMENTS = heapless::consts::U16;
#[allow(non_camel_case_types)]
#[cfg(target_pointer_width = "64")]
type MAX_ELEMENTS = heapless::consts::U8;

use heapless::Vec;

pub struct HeaplessListBuilder<'a, A: StackAlloc> {
    stack: &'a mut A,
    elements: Vec<Term, MAX_ELEMENTS>,
    failed: bool,
}
impl<'a, A: StackAlloc> HeaplessListBuilder<'a, A> {
    /// Creates a new list builder that allocates on the given processes' heap
    #[inline]
    pub fn new(stack: &'a mut A) -> Self {
        Self {
            stack,
            elements: Vec::new(),
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
        assert!(
            term.is_runtime(),
            "invalid list element for stack-allocated list"
        );
        if self.failed {
            self
        } else {
            match self.elements.push(term) {
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
    pub fn finish(self) -> Result<Term, Alloc> {
        if self.failed {
            Err(alloc!())
        } else {
            let size = self.elements.len();
            if size == 0 {
                unsafe {
                    let ptr = self.stack.alloca(1)?.as_ptr();
                    ptr::write(ptr, Term::NIL);
                }
                Ok(Term::NIL)
            } else {
                // Construct allocation layout for list
                let (elements_layout, _) = Layout::new::<Cons>().repeat(size).unwrap();
                let (layout, _) = Layout::new::<Term>().extend(elements_layout).unwrap();
                // Allocate on stack
                let ptr = unsafe { self.stack.alloca_layout(layout)?.as_ptr() };
                // Get pointer to first cell
                let first_ptr = unsafe { ptr.add(1) as *mut Cons };
                // Write header with pointer to first cell
                unsafe { ptr::write(ptr, Term::make_list(first_ptr)) };
                // For each element in the list, write a cell with a pointer to the next one
                for (i, element) in self.elements.iter().copied().enumerate() {
                    // Offsets are relative to the first cell, first element has `i` of 0
                    let cell_ptr = unsafe { first_ptr.add(i) };
                    // Get mutable reference to cell memory
                    let mut cell = unsafe { &mut *cell_ptr };
                    if i < size - 1 {
                        // If we have future cells to write, generate a valid tail
                        let tail_ptr = unsafe { cell_ptr.add(1) };
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
    pub fn finish_with(mut self, term: Term) -> Result<Term, Alloc> {
        if self.failed {
            Err(alloc!())
        } else {
            let size = self.elements.len() + 1;
            if size == 1 {
                // Proper list
                self.elements.push(term).unwrap();
                self.finish()
            } else {
                // Construct allocation layout for list
                let (elements_layout, _) = Layout::new::<Cons>().repeat(size).unwrap();
                let (layout, _) = Layout::new::<Term>().extend(elements_layout).unwrap();
                // Allocate on stack
                let ptr = unsafe { self.stack.alloca_layout(layout)?.as_ptr() };
                // Get pointer to first cell
                let first_ptr = unsafe { ptr.add(1) as *mut Cons };
                // Write header with pointer to first cell
                unsafe { ptr::write(ptr, Term::make_list(first_ptr)) };
                // For each element in the list, write a cell with a pointer to the next one
                for (i, element) in self.elements.iter().copied().enumerate() {
                    // Offsets are relative to the first cell, first element has `i` of 0
                    let cell_ptr = unsafe { first_ptr.add(i) };
                    // Get mutable reference to cell memory
                    let mut cell = unsafe { &mut *cell_ptr };
                    if i < size - 2 {
                        // If we have future cells to write, generate a valid tail
                        let tail_ptr = unsafe { cell_ptr.add(1) };
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
    use crate::erts::term::SmallInteger;

    mod clone_to_heap {
        use super::*;

        use ::alloc::sync::Arc;

        use crate::erts::process::{alloc, Priority, ProcessControlBlock};
        use crate::erts::scheduler;
        use crate::erts::term::{atom_unchecked, Atom};
        use crate::erts::ModuleFunctionArity;

        #[test]
        fn single_element() {
            let process = process();
            let head = atom_unchecked("head");
            let tail = Term::NIL;
            let cons_term = process.cons(head, tail).unwrap();
            let cons: Boxed<Cons> = cons_term.try_into().unwrap();

            let (heap_fragment_cons_term, _) = cons.clone_to_fragment().unwrap();

            assert_eq!(heap_fragment_cons_term, cons_term);
        }

        #[test]
        fn two_element_proper() {
            let process = process();
            let element0 = process.integer(0).unwrap();
            let element1 = process.integer(1).unwrap();
            let tail = Term::NIL;
            let cons_term = process
                .cons(element0, process.cons(element1, tail).unwrap())
                .unwrap();
            let cons: Boxed<Cons> = cons_term.try_into().unwrap();

            let (heap_fragment_cons_term, _) = cons.clone_to_fragment().unwrap();

            assert_eq!(heap_fragment_cons_term, cons_term);
        }

        #[test]
        fn two_element_improper() {
            let process = process();
            let head = atom_unchecked("head");
            let tail = atom_unchecked("tail");
            let cons_term = process.cons(head, tail).unwrap();
            let cons: Boxed<Cons> = cons_term.try_into().unwrap();

            let (heap_fragment_cons_term, _) = cons.clone_to_fragment().unwrap();

            assert_eq!(heap_fragment_cons_term, cons_term);
        }

        fn process() -> ProcessControlBlock {
            let init = Atom::try_from_str("init").unwrap();
            let initial_module_function_arity = Arc::new(ModuleFunctionArity {
                module: init,
                function: init,
                arity: 0,
            });
            let (heap, heap_size) = alloc::default_heap().unwrap();

            let process = ProcessControlBlock::new(
                Priority::Normal,
                None,
                initial_module_function_arity,
                heap,
                heap_size,
            );

            process.schedule_with(scheduler::id::next());

            process
        }
    }

    mod into_iter {
        use super::*;

        #[test]
        fn proper_list_iter() {
            let a = unsafe { SmallInteger::new(42).unwrap().as_term() };
            let b = unsafe { SmallInteger::new(24).unwrap().as_term() };
            let c = unsafe { SmallInteger::new(11).unwrap().as_term() };
            let last = Cons::new(c, Term::NIL);
            let last_term = Term::make_list(&last);
            let second = Cons::new(b, last_term);
            let second_term = Term::make_list(&second);
            let first = Cons::new(a, second_term);

            let mut list_iter = first.into_iter();
            let l0 = list_iter.next().unwrap();
            assert_eq!(l0, Ok(a));
            let l1 = list_iter.next().unwrap();
            assert_eq!(l1, Ok(b));
            let l2 = list_iter.next().unwrap();
            assert_eq!(l2, Ok(c));
            assert_eq!(list_iter.next(), None);
            assert_eq!(list_iter.next(), None);
        }

        #[test]
        fn improper_list_iter() {
            let a = unsafe { SmallInteger::new(42).unwrap().as_term() };
            let b = unsafe { SmallInteger::new(24).unwrap().as_term() };
            let c = unsafe { SmallInteger::new(11).unwrap().as_term() };
            let d = unsafe { SmallInteger::new(99).unwrap().as_term() };
            let last = Cons::new(c, d);
            let last_term = Term::make_list(&last);
            let second = Cons::new(b, last_term);
            let second_term = Term::make_list(&second);
            let first = Cons::new(a, second_term);

            let mut list_iter = first.into_iter();
            let l0 = list_iter.next().unwrap();
            assert_eq!(l0, Ok(a));
            let l1 = list_iter.next().unwrap();
            assert_eq!(l1, Ok(b));
            let l2 = list_iter.next().unwrap();
            assert_eq!(l2, Ok(c));
            let l3 = list_iter.next().unwrap();
            assert_eq!(l3, Err(ImproperList { tail: d }));
            assert_eq!(list_iter.next(), None);
            assert_eq!(list_iter.next(), None);
        }
    }
}

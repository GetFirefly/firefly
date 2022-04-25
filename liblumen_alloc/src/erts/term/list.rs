use core::alloc::Layout;
use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display, Write};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem;
use core::ptr;

use anyhow::*;
use heapless::Vec;
use thiserror::Error;

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::{StackAlloc, TermAlloc};
use crate::erts::term::prelude::*;
use crate::erts::to_word_size;

pub fn optional_cons_to_term(cons: Option<Boxed<Cons>>) -> Term {
    match cons {
        None => Term::NIL,
        Some(boxed) => boxed.into(),
    }
}

pub enum List {
    Empty,
    NonEmpty(Boxed<Cons>),
}

impl TryFrom<Term> for List {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<List, TypeError> {
        term.decode().unwrap().try_into()
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
    pub fn iter(&self) -> self::Iter<'_> {
        Iter {
            head: Some(Ok(self.head)),
            tail: Some(self.tail),
            _marker: PhantomData,
        }
    }

    /// The number of bytes for the header and immediate terms or box term pointer to elements
    /// allocated elsewhere.
    pub fn need_in_bytes_from_len(len: usize) -> usize {
        mem::size_of::<Cons>() * len
    }

    /// The number of words for the header and immediate terms or box term pointer to elements
    /// allocated elsewhere.
    pub fn need_in_words_from_len(len: usize) -> usize {
        to_word_size(Self::need_in_bytes_from_len(len))
    }

    /// Create a new cons cell from a head and tail pointer pair
    #[inline]
    pub fn new(head: Term, tail: Term) -> Self {
        Self { head, tail }
    }

    pub fn contains(&self, term: Term) -> bool {
        self.iter().any(|result| match result {
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

    /// Returns true if this cons is the head of
    ///a proper list.
    pub fn is_proper(&self) -> bool {
        self.iter().all(|result| result.is_ok())
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
            let tail: *mut Cons = self.tail.dyn_cast();
            return Some(MaybeImproper::Proper(unsafe { *tail }));
        }
        Some(MaybeImproper::Improper(self.tail))
    }

    /// Searches this keyword list for the first element which has a matching key
    /// at the given index.
    ///
    /// If no key is found, returns 'badarg'
    pub fn keyfind<I>(&self, index: I, key: Term) -> anyhow::Result<Option<Term>>
    where
        I: TupleIndex + Copy,
    {
        for result in self.iter() {
            if let Ok(item) = result {
                let tuple_item: Result<Boxed<Tuple>, _> = item.try_into();
                if let Ok(tuple) = tuple_item {
                    if let Ok(candidate) = tuple.get_element(index) {
                        if candidate == key {
                            return Ok(Some(item));
                        }
                    }
                }
            } else {
                return Err(ImproperListError.into());
            }
        }

        Ok(None)
    }

    // See https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L117-L140
    fn is_printable_string(&self) -> bool {
        self.iter().all(|result| match result {
            Ok(ref element) => {
                // See https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L128-L129
                let result_char: Result<char, _> = (*element).try_into();

                match result_char {
                    Ok(c) => {
                        // https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L132
                        c.is_ascii_graphic() || c.is_ascii_whitespace()
                    }
                    _ => false,
                }
            }
            _ => false,
        })
    }
}

impl Debug for Cons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;

        let mut iter = self.iter();

        if let Some(first_result) = iter.next() {
            write!(f, "{:?}", first_result.unwrap())?;

            for result in iter {
                match result {
                    Ok(element) => write!(f, ", {:?}", element)?,
                    Err(ImproperList { tail }) => write!(f, " | {:?}", tail)?,
                }
            }
        }

        f.write_char(']')
    }
}

impl Display for Cons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // See https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L423-443
        if self.is_printable_string() {
            f.write_char('\"')?;

            for result in self.iter() {
                // `is_printable_string` guarantees all Ok
                let element = result.unwrap();
                let c: char = element.try_into().unwrap();

                match c {
                    '\n' => f.write_str("\\\n")?,
                    '\"' => f.write_str("\\\"")?,
                    _ => f.write_char(c)?,
                }
            }

            f.write_char('\"')
        } else {
            f.write_char('[')?;

            let mut iter = self.iter();

            if let Some(first_result) = iter.next() {
                write!(f, "{}", first_result.unwrap())?;

                for result in iter {
                    match result {
                        Ok(element) => write!(f, ", {}", element)?,
                        Err(ImproperList { tail }) => write!(f, " | {}", tail)?,
                    }
                }
            }

            f.write_char(']')
        }
    }
}

impl Eq for Cons {}
impl Ord for Cons {
    fn cmp(&self, other: &Cons) -> cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}
impl PartialEq for Cons {
    fn eq(&self, other: &Cons) -> bool {
        self.head.eq(&other.head) && self.tail.eq(&other.tail)
    }
}
impl<T> PartialEq<Boxed<T>> for Cons
where
    T: PartialEq<Cons>,
{
    #[inline]
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}

impl PartialOrd for Cons {
    fn partial_cmp(&self, other: &Cons) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> PartialOrd<Boxed<T>> for Cons
where
    T: PartialOrd<Cons>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<cmp::Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}

impl CloneToProcess for Cons {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        let mut vec: std::vec::Vec<Term> = Default::default();
        let mut tail = Term::NIL;

        for result in self.iter() {
            match result {
                Ok(element) => vec.push(element.clone_to_heap(heap)?),
                Err(ImproperList {
                    tail: improper_tail,
                }) => tail = improper_tail,
            }
        }

        heap.improper_list_from_slice(&vec, tail)
            .map_err(From::from)
            .map(From::from)
    }

    fn size_in_words(&self) -> usize {
        let mut elements = 0;
        let mut words = 0;

        for result in self.iter() {
            let element = match result {
                Ok(element) => element,
                Err(ImproperList { tail }) => tail,
            };

            elements += 1;
            words += element.size_in_words();
        }

        words += Self::need_in_words_from_len(elements);
        words
    }
}

impl<'a> IntoIterator for &'a Cons {
    type Item = Result<Term, ImproperList>;
    type IntoIter = Iter<'a>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ImproperList {
    pub tail: Term,
}

pub struct Iter<'a> {
    head: Option<Result<Term, ImproperList>>,
    tail: Option<Term>,
    _marker: PhantomData<&'a Term>,
}

impl FusedIterator for Iter<'_> {}

impl Iterator for Iter<'_> {
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

                match tail.decode().unwrap() {
                    TypedTerm::Nil => {
                        self.head = None;
                        self.tail = None;
                    }
                    TypedTerm::List(cons_ptr) => {
                        let cons = cons_ptr.as_ref();
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

#[derive(Debug, Error)]
#[error("improper list")]
pub struct ImproperListError;

impl TryFrom<TypedTerm> for Boxed<Cons> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::List(cons) => Ok(cons),
            _ => Err(TypeError),
        }
    }
}

impl TryInto<String> for Boxed<Cons> {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<String, Self::Error> {
        self.iter()
            .map(|result| match result {
                Ok(element) => {
                    let result_char: Result<char, _> = element
                        .try_into()
                        .context("string (Erlang) or charlist (elixir) element not a char");

                    result_char
                }
                Err(_) => Err(ImproperListError.into()),
            })
            .collect()
    }
}

pub struct ListBuilder<'a, A: TermAlloc> {
    heap: &'a mut A,
    element: Option<Term>,
    first: *mut Cons,
    current: *mut Cons,
    size: usize,
    failed: bool,
}
impl<'a, A: TermAlloc> ListBuilder<'a, A> {
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
    fn push_internal(&mut self, term: Term) -> AllocResult<()> {
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
                current.tail = new_current.into();
                self.current = new_current;
                self.size += 1;
            }
        }
        Ok(())
    }

    /// Consumes the builder and produces a `Term` which points to the allocated list
    #[inline]
    pub fn finish(mut self) -> AllocResult<Boxed<Cons>> {
        if self.failed {
            // We can't clean up the heap until GC
            Err(alloc!())
        } else {
            match self.element {
                // Empty list
                None => {
                    assert!(self.first.is_null());
                    // Allocate an empty cell for the list
                    // Technically this is a bit wasteful, since we could just use NIL
                    // as an immediate, but to return a `Boxed<Cons>` we need a value to
                    // point to
                    let first_ptr = self.alloc_cell(Term::NIL, Term::NIL)?;
                    Ok(unsafe { Boxed::new_unchecked(first_ptr) })
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
                        current.tail = new_current.into();
                    } else {
                        // This occurs when we've filled a cell, so we need to allocate two cells,
                        // one for the current tail and one for the last element
                        let cadr_cell = self.alloc_cell(last, Term::NIL)?;
                        let cadr = cadr_cell.into();
                        let cdr = self.alloc_cell(current.tail, cadr)?;
                        current.tail = cdr.into();
                    }
                    // Return a reference to the first cell of the list
                    Ok(unsafe { Boxed::new_unchecked(self.first) })
                }
            }
        }
    }

    /// Like `finish`, but produces an improper list if there is already at least one element
    #[inline]
    pub fn finish_with(mut self, term: Term) -> AllocResult<Boxed<Cons>> {
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
                        current.tail = cdr.into();
                    } else {
                        let tail = self.clone_term_to_heap(term)?;
                        let cadr_cell = self.alloc_cell(next, tail)?;
                        let cadr = cadr_cell.into();
                        let cdr = self.alloc_cell(current.tail, cadr)?;
                        current.tail = cdr.into();
                    }
                    Ok(unsafe { Boxed::new_unchecked(self.first) })
                }
            }
        }
    }

    #[inline]
    fn clone_term_to_heap(&mut self, term: Term) -> AllocResult<Term> {
        if term.is_immediate() {
            Ok(term)
        } else if term.is_boxed() {
            let ptr: *mut Term = term.dyn_cast();
            if !term.is_literal() && self.heap.is_owner(ptr) {
                // No need to clone
                Ok(term)
            } else {
                term.clone_to_heap(self.heap)
            }
        } else if term.is_non_empty_list() {
            let ptr: *mut Cons = term.dyn_cast();
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
    fn alloc_cell(&mut self, head: Term, tail: Term) -> AllocResult<*mut Cons> {
        let layout = Layout::new::<Cons>();
        let ptr = unsafe { self.heap.alloc_layout(layout)?.as_ptr() as *mut Cons };
        let mut cell = unsafe { &mut *ptr };
        cell.head = head;
        cell.tail = tail;
        Ok(ptr)
    }
}

#[allow(non_camel_case_types)]
#[cfg(target_pointer_width = "32")]
const MAX_ELEMENTS: usize = 16;
#[allow(non_camel_case_types)]
#[cfg(target_pointer_width = "64")]
const MAX_ELEMENTS: usize = 8;

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
            term.is_valid(),
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
    pub fn finish(self) -> AllocResult<Boxed<Cons>> {
        if self.failed {
            Err(alloc!())
        } else {
            let size = self.elements.len();
            if size == 0 {
                unsafe {
                    let ptr = self.stack.alloca(1)?.as_ptr();
                    ptr::write(ptr, Term::NIL);
                    Ok(Boxed::new_unchecked(ptr as *mut Cons))
                }
            } else {
                // Construct allocation layout for list
                let (elements_layout, _) = Layout::new::<Cons>().repeat(size).unwrap();
                let (layout, _) = Layout::new::<Term>().extend(elements_layout).unwrap();
                // Allocate on stack
                let ptr = unsafe { self.stack.alloca_layout(layout)?.as_ptr() };
                // Get pointer to first cell
                let first_ptr = unsafe { ptr.add(1) as *mut Cons };
                // Write header with pointer to first cell
                unsafe { ptr::write(ptr, first_ptr.into()) };
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
                        cell.tail = tail_ptr.into();
                    } else {
                        // This is the last element
                        cell.head = element;
                        cell.tail = Term::NIL;
                    }
                }
                Ok(unsafe { Boxed::new_unchecked(first_ptr) })
            }
        }
    }

    /// Like `finish`, but produces an improper list if there is already at least one element
    #[inline]
    pub fn finish_with(mut self, term: Term) -> AllocResult<Boxed<Cons>> {
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
                unsafe { ptr::write(ptr, first_ptr.into()) };
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
                        cell.tail = tail_ptr.into();
                    } else {
                        // This is the last cell
                        cell.head = element;
                        cell.tail = term;
                    }
                }
                Ok(unsafe { Boxed::new_unchecked(first_ptr) })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::erts::testing::RegionHeap;

    mod clone_to_heap {
        use super::*;

        #[test]
        fn list_single_element() {
            let mut heap = RegionHeap::default();

            let cons = cons!(heap, atom!("head"));

            let cloned_term = cons.clone_to_heap(&mut heap).unwrap();
            let cloned: Boxed<Cons> = cloned_term.try_into().unwrap();
            assert_eq!(cons, cloned);
        }

        #[test]
        fn list_two_element_proper() {
            let mut heap = RegionHeap::default();
            let element0 = fixnum!(0);
            let element1 = fixnum!(1);

            let cons = cons!(heap, element0, element1);

            let cloned_term = cons.clone_to_heap(&mut heap).unwrap();
            let cloned: Boxed<Cons> = cloned_term.try_into().unwrap();
            assert_eq!(cons, cloned);
        }

        #[test]
        fn list_three_element_proper() {
            let mut heap = RegionHeap::default();
            let element0 = fixnum!(0);
            let element1 = fixnum!(1);
            let element2 = heap.binary_from_str("hello world!").unwrap();

            let cons = cons!(heap, element0, element1, element2);

            let cloned_term = cons.clone_to_heap(&mut heap).unwrap();
            let cloned: Boxed<Cons> = cloned_term.try_into().unwrap();
            assert_eq!(cons, cloned);
        }

        #[test]
        fn list_three_element_proper_with_nil_element() {
            let mut heap = RegionHeap::default();
            let element0 = fixnum!(0);
            let element1 = Term::NIL;
            let element2 = fixnum!(2);

            let cons = cons!(heap, element0, element1, element2);

            let cloned_term = cons.clone_to_heap(&mut heap).unwrap();
            let cloned: Boxed<Cons> = cloned_term.try_into().unwrap();
            assert_eq!(cons, cloned);
        }

        #[test]
        fn list_two_element_improper() {
            let mut heap = RegionHeap::default();
            let head = atom!("head");
            let tail = atom!("tail");

            let cons = improper_cons!(heap, head, tail);

            let cloned_term = cons.clone_to_heap(&mut heap).unwrap();
            let cloned: Boxed<Cons> = cloned_term.try_into().unwrap();
            assert_eq!(cons, cloned);
        }
    }

    mod into_iter {
        use super::*;

        #[test]
        fn list_proper_list_iter() {
            let mut heap = RegionHeap::default();
            let a = fixnum!(42);
            let b = fixnum!(24);
            let c = fixnum!(11);
            let cons = cons!(heap, a, b, c);

            let mut list_iter = cons.into_iter();
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
        fn list_improper_list_iter() {
            let mut heap = RegionHeap::default();
            let a = fixnum!(42);
            let b = fixnum!(24);
            let c = fixnum!(11);
            let d = fixnum!(99);
            let cons = improper_cons!(heap, a, b, c, d);

            let mut list_iter = cons.into_iter();
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

    mod builder {
        use super::*;

        #[test]
        fn list_builder_proper_list_iter() {
            let mut heap = RegionHeap::default();
            let a = fixnum!(42);
            let b = fixnum!(24);
            let c = fixnum!(11);
            let cons = ListBuilder::new(&mut heap)
                .push(a)
                .push(b)
                .push(c)
                .finish()
                .unwrap();

            let mut list_iter = cons.into_iter();
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
        fn list_builder_improper_list_iter() {
            let mut heap = RegionHeap::default();
            let a = fixnum!(42);
            let b = fixnum!(24);
            let c = fixnum!(11);
            let d = fixnum!(99);
            let cons = ListBuilder::new(&mut heap)
                .push(a)
                .push(b)
                .push(c)
                .finish_with(d)
                .unwrap();

            let mut list_iter = cons.into_iter();
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

use core::alloc::Layout;
use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display, Write};
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::mem;
use core::ops::Deref;

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::term::index::{self, try_from_one_based_term_to_zero_based_usize};

use super::*;

/// Represents a tuple term in memory.
///
/// The size is also the header for the term, but in the
/// case of tuples, there are not any bits actually set, as
/// its the only term with an arityval of zero.
///
/// The `head` pointer is a pointer to the first element in the tuple,
/// typically we will construct `Tuple` like a `Vec<Term>`, followed by
/// any elements that are not allocated elsewhere, so we can keep things
/// in the same cache line when possible; but this is not strictly required,
/// as we still have to follow pointers to get at the individual elements,
/// so whether they are right next to the `Tuple` itself, or elsewhere is not
/// critical
#[repr(C)]
pub struct Tuple {
    header: Term,
}
impl Tuple {
    pub fn clone_to_heap_from_elements<A: HeapAlloc>(
        heap: &mut A,
        elements: &[Term],
    ) -> Result<Term, Alloc> {
        // The result of calling this will be a Tuple with everything located
        // contiguously in memory
        unsafe {
            // Allocate the space needed for the header and all the elements
            let len = elements.len();
            let words = Self::need_in_words_from_len(len);

            let tuple_ptr = heap.alloc(words)?.as_ptr() as *mut Self;
            // Write the header
            tuple_ptr.write(Tuple::new(len));

            let mut element_ptr = tuple_ptr.offset(1) as *mut Term;
            // Write each element
            for element in elements {
                if element.is_immediate() {
                    element_ptr.write(*element);
                } else {
                    // Recursively call clone_to_heap, and then write the box header here
                    let boxed = element.clone_to_heap(heap)?;
                    element_ptr.write(boxed);
                }

                element_ptr = element_ptr.offset(1);
            }

            Ok(Term::make_boxed(tuple_ptr))
        }
    }

    /// Create a new `Tuple` struct
    ///
    /// NOTE: This does not allocate space for the tuple, it simply
    /// constructs an instance of the `Tuple` header, other functions
    /// can then use this in conjunction with a `Layout` for the elements
    /// to allocate the appropriate amount of memory
    #[inline]
    pub fn new(size: usize) -> Self {
        Self {
            header: Term::make_header(size, Term::FLAG_TUPLE),
        }
    }

    /// Returns the length of this tuple
    #[inline]
    pub fn len(&self) -> usize {
        self.header.arityval()
    }

    /// Returns a pointer to the head element
    ///
    /// NOTE: This is unsafe to use unless you know the tuple has been allocated
    #[inline]
    pub fn head(&self) -> *mut Term {
        unsafe { (self as *const Tuple).add(1) as *mut Term }
    }

    /// This function produces a `Layout` which represents the memory layout
    /// needed for the tuple header, and `num_elements` terms. The resulting
    /// size is only enough for the tuple and word-sized values, e.g. immediates
    /// or boxes. You need to extend this layout with others representing more
    /// complex values like maps/lists/etc., if you want a layout that covers all
    /// the memory needed by elements of the tuple
    #[inline]
    pub fn layout(num_elements: usize) -> Layout {
        let size = Self::need_in_bytes_from_len(num_elements);
        unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<Term>()) }
    }

    /// The number of bytes for the header and immediate terms or box term pointer to elements
    /// allocated elsewhere.
    pub fn need_in_bytes_from_len(len: usize) -> usize {
        mem::size_of::<Self>() + (len * mem::size_of::<Term>())
    }

    /// The number of words for the header and immediate terms or box term pointer to elements
    /// allocated elsewhere.
    pub fn need_in_words_from_len(len: usize) -> usize {
        to_word_size(Self::need_in_bytes_from_len(len))
    }

    /// The number of words for the header and immediate terms or box term pointer and the data
    /// the box is pointing to.
    pub fn need_in_words_from_elements(elements: &[Term]) -> usize {
        let mut words = Self::need_in_words_from_len(elements.len());

        for element in elements {
            words += element.size_in_words();
        }

        words
    }

    /// Constructs an iterator over elements of the tuple
    #[inline]
    pub fn iter(&self) -> Iter {
        Iter::new(self)
    }

    /// Sets an element in the tuple, returns `Ok(())` if successful,
    /// otherwise returns `Err(IndexErr)` if the given index is invalid
    pub fn set_element_from_one_based_term_index(
        &mut self,
        index: Term,
        element: Term,
    ) -> Result<(), index::Error> {
        let index_usize: usize = try_from_one_based_term_to_zero_based_usize(index)?;

        self.set_element_from_zero_based_usize_index(index_usize, element)
    }

    /// Like `set_element` but for internal runtime use, as it takes a zero-based `usize` directly
    pub fn set_element_from_zero_based_usize_index(
        &mut self,
        index: usize,
        element: Term,
    ) -> Result<(), index::Error> {
        let len = self.len();

        if index < len {
            unsafe {
                self.head().add(index).write(element);

                Ok(())
            }
        } else {
            Err(index::Error::OutOfBounds { len, index })
        }
    }

    /// Fetches an element from the tuple, returns `Ok(term)` if the index is
    /// valid, otherwise returns `Err(IndexErr)`
    pub fn get_element_from_one_based_term_index(&self, index: Term) -> Result<Term, index::Error> {
        let index_usize: usize = try_from_one_based_term_to_zero_based_usize(index)?;

        self.get_element_from_zero_based_usize_index(index_usize)
    }

    /// Like `get_element` but for internal runtime use, as it takes a `usize`
    /// directly, rather than a value of type `Term`
    pub fn get_element_from_zero_based_usize_index(
        &self,
        index: usize,
    ) -> Result<Term, index::Error> {
        let len = self.len();

        if index < len {
            unsafe {
                let ptr = self.head().add(index);

                Ok(follow_moved(*ptr))
            }
        } else {
            Err(index::Error::OutOfBounds { index, len })
        }
    }
}

unsafe impl AsTerm for Tuple {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self)
    }
}

impl CloneToProcess for Tuple {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        Tuple::clone_to_heap_from_elements(heap, self)
    }

    fn size_in_words(&self) -> usize {
        Self::need_in_words_from_elements(self)
    }
}

impl Debug for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut debug_tuple = f.debug_tuple("Tuple");
        let mut debug_tuple_ref = &mut debug_tuple;

        for element in self.iter() {
            debug_tuple_ref = debug_tuple_ref.field(&element);
        }

        debug_tuple_ref.finish()
    }
}

impl Deref for Tuple {
    type Target = [Term];

    fn deref(&self) -> &[Term] {
        unsafe { core::slice::from_raw_parts(self.head(), self.len()) }
    }
}

impl Display for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('{')?;

        let mut iter = self.iter();

        if let Some(first_element) = iter.next() {
            write!(f, "{}", first_element)?;

            for element in iter {
                write!(f, ", {}", element)?;
            }
        }

        f.write_char('}')
    }
}

impl Hash for Tuple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for element in self.iter() {
            element.hash(state);
        }
    }
}

impl PartialEq for Tuple {
    fn eq(&self, other: &Tuple) -> bool {
        self.iter().eq(other.iter())
    }
}
impl PartialOrd for Tuple {
    fn partial_cmp(&self, other: &Tuple) -> Option<cmp::Ordering> {
        use core::cmp::Ordering;

        match self.len().cmp(&other.len()) {
            Ordering::Less => return Some(Ordering::Less),
            Ordering::Greater => return Some(Ordering::Greater),
            Ordering::Equal => self.iter().partial_cmp(other.iter()),
        }
    }
}

impl TryFrom<Term> for Boxed<Tuple> {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Boxed<Tuple> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Boxed(boxed_tuple) => boxed_tuple.to_typed_term().unwrap().try_into(),
            TypedTerm::Tuple(tuple) => Ok(tuple),
            _ => Err(TypeError),
        }
    }
}

pub struct Iter {
    pointer: *const Term,
    limit: *const Term,
}

impl Iter {
    pub fn new(tuple: &Tuple) -> Self {
        let pointer = tuple.head();
        let limit = unsafe { pointer.add(tuple.len()) };

        Self { pointer, limit }
    }
}

impl DoubleEndedIterator for Iter {
    fn next_back(&mut self) -> Option<Term> {
        if self.pointer == self.limit {
            None
        } else {
            unsafe {
                // limit is +1 past he actual elements, so pre-decrement unlike `next`, which
                // post-decrements
                self.limit = self.limit.offset(-1);
                self.limit.as_ref().map(|r| *r)
            }
        }
    }
}

impl Iterator for Iter {
    type Item = Term;

    fn next(&mut self) -> Option<Term> {
        if self.pointer == self.limit {
            None
        } else {
            let old_pointer = self.pointer;

            unsafe {
                self.pointer = self.pointer.add(1);
                old_pointer.as_ref().map(|r| *r)
            }
        }
    }
}

impl FusedIterator for Iter {}

#[cfg(test)]
mod tests {
    use super::*;

    use core::convert::TryInto;

    use alloc::sync::Arc;

    use crate::erts::process::{default_heap, Priority, ProcessControlBlock};
    use crate::erts::scheduler;
    use crate::erts::term::{Boxed, Tuple};
    use crate::erts::ModuleFunctionArity;

    mod get_element_from_zero_based_usize_index {
        use super::*;

        #[test]
        fn without_valid_index() {
            let process = process();
            let tuple_term = process.tuple_from_slice(&[]).unwrap();
            let boxed_tuple: Boxed<Tuple> = tuple_term.try_into().unwrap();

            assert_eq!(
                boxed_tuple.get_element_from_zero_based_usize_index(1),
                Err(index::Error::OutOfBounds { index: 1, len: 0 })
            );
        }

        #[test]
        fn with_valid_index() {
            let process = process();
            let tuple_term = process
                .tuple_from_slice(&[process.integer(0).unwrap()])
                .unwrap();
            let boxed_tuple: Boxed<Tuple> = tuple_term.try_into().unwrap();

            assert_eq!(
                boxed_tuple.get_element_from_zero_based_usize_index(0),
                Ok(process.integer(0).unwrap())
            );
        }
    }

    mod eq {
        use super::*;

        #[test]
        fn without_element() {
            let process = process();
            let tuple = process.tuple_from_slice(&[]).unwrap();
            let equal = process.tuple_from_slice(&[]).unwrap();

            assert_eq!(tuple, tuple);
            assert_eq!(tuple, equal);
        }

        #[test]
        fn with_unequal_length() {
            let process = process();
            let tuple = process
                .tuple_from_slice(&[process.integer(0).unwrap()])
                .unwrap();
            let unequal = process
                .tuple_from_slice(&[process.integer(0).unwrap(), process.integer(1).unwrap()])
                .unwrap();

            assert_ne!(tuple, unequal);
        }
    }

    mod iter {
        use super::*;

        #[test]
        fn without_elements() {
            let process = process();
            let tuple_term = process.tuple_from_slice(&[]).unwrap();
            let boxed_tuple: Boxed<Tuple> = tuple_term.try_into().unwrap();

            assert_eq!(boxed_tuple.iter().count(), 0);

            let length = boxed_tuple.len();

            assert_eq!(boxed_tuple.iter().count(), length);
        }

        #[test]
        fn with_elements() {
            let process = process();
            // one of every type
            let slice = &[
                // small integer
                process.integer(0).unwrap(),
                // big integer
                process.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
                process.reference(0).unwrap(),
                closure(&process),
                process.float(0.0).unwrap(),
                process.external_pid_with_node_id(1, 0, 0).unwrap(),
                Term::NIL,
                make_pid(0, 0).unwrap(),
                atom_unchecked("atom"),
                process.tuple_from_slice(&[]).unwrap(),
                process.map_from_slice(&[]).unwrap(),
                process.list_from_slice(&[]).unwrap(),
            ];
            let tuple_term = process.tuple_from_slice(slice).unwrap();
            let boxed_tuple: Boxed<Tuple> = tuple_term.try_into().unwrap();

            assert_eq!(boxed_tuple.iter().count(), 12);

            let length = boxed_tuple.len();

            assert_eq!(boxed_tuple.iter().count(), length);
        }
    }

    mod len {
        use super::*;

        #[test]
        fn without_elements() {
            let tuple = Tuple::new(0);

            assert_eq!(tuple.len(), 0);
        }

        #[test]
        fn with_elements() {
            let tuple = Tuple::new(1);

            assert_eq!(tuple.len(), 1);
        }
    }

    fn closure(process: &ProcessControlBlock) -> Term {
        let creator = process.pid_term();

        let module = Atom::try_from_str("module").unwrap();
        let function = Atom::try_from_str("function").unwrap();
        let arity = 0;
        let module_function_arity = Arc::new(ModuleFunctionArity {
            module,
            function,
            arity,
        });
        let code = |arc_process: &Arc<ProcessControlBlock>| {
            arc_process.wait();

            Ok(())
        };

        process
            .acquire_heap()
            .closure_with_env_from_slices(module_function_arity, code, creator, &[&[]])
            .unwrap()
    }

    fn process() -> ProcessControlBlock {
        let init = Atom::try_from_str("init").unwrap();
        let initial_module_function_arity = Arc::new(ModuleFunctionArity {
            module: init,
            function: init,
            arity: 0,
        });
        let (heap, heap_size) = default_heap().unwrap();

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

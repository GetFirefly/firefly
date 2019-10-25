use core::alloc::Layout;
use core::cmp;
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display, Write};
use core::hash::{Hash, Hasher};
use core::ops::Deref;
use core::slice;
use core::ptr;

use crate::borrow::CloneToProcess;
use crate::erts::{self, HeapAlloc};
use crate::erts::exception::system::Alloc;

use super::prelude::*;
use super::index::{OneBasedIndex, ZeroBasedIndex, IndexError};

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
    header: Header<Tuple>,
    elements: [Term]
}
impl Tuple {
    /// Constructs a new `Tuple` of size `len` using `heap`
    ///
    /// The constructed tuple will contain invalid words until
    /// individual elements are written, this is intended for
    /// cases where we don't already have a slice of elements
    /// to construct a tuple from
    pub fn new<A>(heap: &mut A, len: usize) -> Result<Boxed<Tuple>, Alloc> 
    where
        A: ?Sized + HeapAlloc,
    {
        let layout = Self::layout_for_len(len);

        let header = Header::from_arity(len);
        unsafe {
            // Allocate space for tuple and immediate elements
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Header<Tuple>;
            // Write tuple header
            ptr.write(header);
            // Construct actual Tuple reference
            Ok(Self::from_raw_parts(ptr as *mut u8, len))
        }
    }

    pub fn from_slice<A>(heap: &mut A, elements: &[Term]) -> Result<Boxed<Tuple>, Alloc> 
    where
        A: ?Sized + HeapAlloc,
    {
        let len = elements.len();
        let (layout, data_offset) = Self::layout_for(elements);

        // The result of calling this will be a Tuple with everything located
        // contiguously in memory
        let header = Header::from_arity(len);
        unsafe {
            // Allocate space for tuple and immediate elements
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Header<Tuple>;
            // Write tuple header
            ptr.write(header);
            // Construct pointer to first data element
            let mut data_ptr = (ptr as *mut u8).offset(data_offset as isize) as *mut Term;
            // Walk original slice of terms and copy them into new memory region,
            // copying boxed terms recursively as necessary
            for element in elements {
                if element.is_immediate() {
                    data_ptr.write(*element);
                } else {
                    // Recursively call clone_to_heap, and then write the box header here
                    let boxed = element.clone_to_heap(heap)?;
                    data_ptr.write(boxed);
                }

                data_ptr = data_ptr.offset(1);
            }
            // Construct actual Tuple reference
            Ok(Self::from_raw_parts(ptr as *mut u8, len))
        }
    }

    #[inline]
    pub unsafe fn from_raw_term(term: *mut Term) -> Boxed<Self> {
        let header = &*(term as *mut Header<Term>);
        let arity = header.arity();

        Self::from_raw_parts(term as *const u8, arity)
    }

    /// Constructs a reference to a `Tuple` given a pointer to
    /// the memory containing the struct and the length of its variable-length
    /// data
    ///
    /// NOTE: For more information about how this works, see the detailed
    /// explanation in the function docs for `HeapBin::from_raw_parts`, the
    /// details are mostly identical, all that differs is the type of data
    #[inline]
    pub(in super) unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Boxed<Self> {
        // Invariants of slice::from_raw_parts.
        assert!(!ptr.is_null());
        assert!(len <= isize::max_value() as usize);

        let slice = core::slice::from_raw_parts(ptr as *const (), len);
        let ptr = slice as *const [()] as *mut Self;
        Boxed::new_unchecked(ptr)
    }


    /// Returns the length of this tuple
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// This function produces a `Layout` which represents the memory layout
    /// needed for the tuple header, and `num_elements` terms. The resulting
    /// size is only enough for the tuple and word-sized values, e.g. immediates
    /// or boxes. You need to extend this layout with others representing more
    /// complex values like maps/lists/etc., if you want a layout that covers all
    /// the memory needed by elements of the tuple

    #[inline]
    pub(in crate::erts) fn layout_for(elements: &[Term]) -> (Layout, usize) {
        let (base_layout, data_offset) = Layout::new::<Header<Tuple>>()
            .extend(Layout::for_value(elements))
            .unwrap();
        // We pad to alignment so that the Layout produced here
        // matches that returned by `Layout::for_value` on the
        // final `HeapBin`
        let layout = base_layout
            .pad_to_align()
            .unwrap();

        (layout, data_offset)
    }

    #[inline]
    pub(in crate::erts) fn layout_for_len(len: usize) -> Layout {
        // Construct a false tuple of size `len` to get an accurate layout
        //
        // NOTE: This is essentially compiler magic; we don't really create
        // anything here, nor dereference the non-existent pointers we crafted,
        // what happens is that we construct a fat pointer that is read by the
        // Layout API in order to calculate a size and alignment for the value
        unsafe {
            let ptr = ptr::null_mut() as *mut Term;
            let arr = core::slice::from_raw_parts(ptr as *const (), len + 1);
            let tup = Boxed::new_unchecked(arr as *const [()] as *const _ as *mut Self);
            Layout::for_value(tup.as_ref())
        }
    }

    /// The number of words for the header and immediate terms or box term pointer and the data
    /// the box is pointing to.
    pub fn need_in_words_from_elements(elements: &[Term]) -> usize {
        let (layout, _) = Self::layout_for(elements);
        let mut words = erts::to_word_size(layout.size());

        for element in elements {
            words += element.size_in_words();
        }

        words
    }

    /// Constructs an iterator over elements of the tuple
    #[inline]
    pub fn iter<'a>(&'a self) -> slice::Iter<'a, Term> {
        self.elements.iter()
    }

    /// Returns a slice containing the tuple elements
    #[inline]
    pub fn elements(&self) -> &[Term] {
        &self.elements
    }

    /// Returns a mutable slice containing the tuple elements
    #[inline]
    pub fn elements_mut(&mut self) -> &mut [Term] {
        &mut self.elements
    }

    /// Sets an element in the tuple, returns `Ok(())` if successful,
    /// otherwise returns `Err(IndexErr)` if the given index is invalid
    pub fn set_element_from_one_based_term_index(
        &mut self,
        index: OneBasedIndex,
        element: Term,
    ) -> Result<(), IndexError> {
        self.set_element_from_zero_based_usize_index(index.into(), element)
    }

    /// Like `set_element` but for internal runtime use, as it takes a zero-based `usize` directly
    #[inline]
    pub fn set_element_from_zero_based_usize_index(
        &mut self,
        index: ZeroBasedIndex,
        element: Term,
    ) -> Result<(), IndexError> {
        let index: usize = index.into();
        if let Some(term) = self.elements.get_mut(index) {
            *term = element;
            return Ok(());
        }

        let len = self.len();
        Err(IndexError::OutOfBounds { len, index })
    }

    /// Fetches an element from the tuple, returns `Ok(term)` if the index is
    /// valid, otherwise returns `Err(IndexErr)`
    pub fn get_element_from_one_based_term_index(&self, index: OneBasedIndex) -> Result<Term, IndexError> {
        self.get_element_from_zero_based_usize_index(index.into())
    }

    /// Like `get_element` but for internal runtime use, as it takes a `usize`
    /// directly, rather than a value of type `Term`
    #[inline]
    pub fn get_element_from_zero_based_usize_index(
        &self,
        index: ZeroBasedIndex,
    ) -> Result<Term, IndexError> {
        let index: usize = index.into();
        if let Some(term) = self.elements.get(index) {
            return Ok(*term);
        }

        Err(IndexError::OutOfBounds { index, len: self.elements.len() })
    }
}

impl CloneToProcess for Tuple {
    fn clone_to_heap<A>(&self, heap: &mut A) -> Result<Term, Alloc> 
    where
        A: ?Sized + HeapAlloc,
    {
        Tuple::from_slice(heap, &self.elements)
            .map(|nn| nn.into())
    }

    fn size_in_words(&self) -> usize {
        erts::to_word_size(Layout::for_value(self).size())
    }
}

impl Debug for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut debug_tuple = f.debug_tuple("Tuple");
        let mut debug_tuple_ref = &mut debug_tuple;

        for element in self.elements.iter() {
            debug_tuple_ref = debug_tuple_ref.field(&element);
        }

        debug_tuple_ref.finish()
    }
}

impl Deref for Tuple {
    type Target = [Term];

    fn deref(&self) -> &[Term] {
        &self.elements
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
        for element in self.elements() {
            element.hash(state);
        }
    }
}

impl PartialEq for Tuple {
    fn eq(&self, other: &Tuple) -> bool {
        self.iter().eq(other.iter())
    }
}
impl<T> PartialEq<Boxed<T>> for Tuple
where
    T: ?Sized + PartialEq<Tuple>,
{
    #[inline]
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
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
impl<T> PartialOrd<Boxed<T>> for Tuple
where
    T: ?Sized + PartialOrd<Tuple>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<cmp::Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}


impl TryFrom<TypedTerm> for Boxed<Tuple> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Tuple(tuple) => Ok(tuple),
            _ => Err(TypeError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::convert::TryInto;

    use alloc::sync::Arc;

    use crate::erts::process::{default_heap, Priority, Process};
    use crate::erts::scheduler;
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
                Err(IndexError::OutOfBounds { index: 1, len: 0 })
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
                Pid::make_term(0, 0).unwrap(),
                Atom::str_to_term("atom"),
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

    fn closure(process: &Process) -> Term {
        let creator = process.pid_term();

        let module = Atom::try_from_str("module").unwrap();
        let function = Atom::try_from_str("function").unwrap();
        let arity = 0;
        let module_function_arity = Arc::new(ModuleFunctionArity {
            module,
            function,
            arity,
        });
        let code = |arc_process: &Arc<Process>| {
            arc_process.wait();

            Ok(())
        };

        process
            .acquire_heap()
            .closure_with_env_from_slices(module_function_arity, code, creator, &[&[]])
            .unwrap()
    }

    fn process() -> Process {
        let init = Atom::try_from_str("init").unwrap();
        let initial_module_function_arity = Arc::new(ModuleFunctionArity {
            module: init,
            function: init,
            arity: 0,
        });
        let (heap, heap_size) = default_heap().unwrap();

        let process = Process::new(
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

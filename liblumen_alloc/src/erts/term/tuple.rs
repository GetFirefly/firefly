use core::alloc::Layout;
use core::cmp;
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display, Write};
use core::hash::{Hash, Hasher};
use core::ptr;
use core::slice;

use crate::borrow::CloneToProcess;
use crate::erts;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::{HeapAlloc, TermAlloc};

use super::prelude::*;

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
    elements: [Term],
}
impl_dynamic_header!(Tuple, Term::HEADER_TUPLE);
impl Tuple {
    /// Constructs a new `Tuple` of size `len` using `heap`
    ///
    /// The constructed tuple will contain invalid words until
    /// individual elements are written, this is intended for
    /// cases where we don't already have a slice of elements
    /// to construct a tuple from
    pub fn new<A>(heap: &mut A, len: usize) -> AllocResult<Boxed<Tuple>>
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
            Ok(Self::from_raw_parts::<Term>(ptr as *mut Term, len))
        }
    }

    pub fn from_slice<A>(heap: &mut A, elements: &[Term]) -> AllocResult<Boxed<Tuple>>
    where
        A: ?Sized + TermAlloc,
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
            Ok(Self::from_raw_parts::<Term>(ptr as *mut Term, len))
        }
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
    pub fn layout_for(elements: &[Term]) -> (Layout, usize) {
        let (base_layout, data_offset) = Layout::new::<Header<Tuple>>()
            .extend(Layout::for_value(elements))
            .unwrap();
        // We pad to alignment so that the Layout produced here
        // matches that returned by `Layout::for_value` on the
        // final `Tuple`
        let layout = base_layout.pad_to_align();

        (layout, data_offset)
    }

    #[inline]
    pub fn layout_for_len(len: usize) -> Layout {
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

    /// Sets the element at the given index
    #[inline]
    pub fn set_element<I: TupleIndex>(
        &mut self,
        index: I,
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

    /// Like `get_element` but for internal runtime use, as it takes a `usize`
    /// directly, rather than a value of type `Term`
    #[inline]
    pub fn get_element<I: TupleIndex>(&self, index: I) -> Result<Term, IndexError> {
        let index: usize = index.into();
        if let Some(term) = self.elements.get(index) {
            return Ok(*term);
        }

        Err(IndexError::OutOfBounds {
            index,
            len: self.elements.len(),
        })
    }

    /// Given a raw pointer to some memory, and a length in units of `Self::Element`,
    /// this function produces a fat pointer to `Self` which represents a value
    /// containing `len` elements in its variable-length field
    ///
    /// For example, given a pointer to the memory containing a `Tuple`, and the
    /// number of elements it contains, this function produces a valid pointer of
    /// type `Tuple`
    unsafe fn from_raw_parts<E: super::arch::Repr>(ptr: *const E, len: usize) -> Boxed<Tuple> {
        // Invariants of slice::from_raw_parts.
        assert!(!ptr.is_null());
        assert!(len <= isize::max_value() as usize);

        let slice = core::slice::from_raw_parts_mut(ptr as *mut E, len);
        let ptr = slice as *mut [E] as *mut Tuple;
        Boxed::new_unchecked(ptr)
    }
}

impl<E: crate::erts::term::arch::Repr> Boxable<E> for Tuple {}

impl<E: super::arch::Repr> UnsizedBoxable<E> for Tuple {
    unsafe fn from_raw_term(ptr: *mut E) -> Boxed<Tuple> {
        let header = &*(ptr as *mut Header<Tuple>);
        let arity = header.arity();

        Self::from_raw_parts::<E>(ptr, arity)
    }
}

impl CloneToProcess for Tuple {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        Tuple::from_slice(heap, &self.elements).map(|nn| nn.into())
    }

    fn size_in_words(&self) -> usize {
        let mut size = erts::to_word_size(Layout::for_value(self).size());
        for element in &self.elements {
            size += element.size_in_words()
        }
        size
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

impl Eq for Tuple {}
impl PartialEq for Tuple {
    #[inline]
    fn eq(&self, other: &Tuple) -> bool {
        self.elements.eq(&other.elements)
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

impl Ord for Tuple {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.elements.cmp(&other.elements)
    }
}
impl PartialOrd for Tuple {
    #[inline]
    fn partial_cmp(&self, other: &Tuple) -> Option<cmp::Ordering> {
        self.elements.partial_cmp(&other.elements)
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

    use alloc::sync::Arc;
    use core::ffi::c_void;

    use crate::erts::testing::RegionHeap;
    use crate::erts::{scheduler, Node};

    mod get_element {
        use super::*;

        #[test]
        fn tuple_zerobased_index_out_of_bounds() {
            let mut heap = RegionHeap::default();
            let tuple = heap.tuple_from_slice(&[]).unwrap();

            assert_eq!(
                tuple.get_element(ZeroBasedIndex::new(1)),
                Err(IndexError::OutOfBounds { index: 1, len: 0 })
            );
        }

        #[test]
        fn tuple_onebased_index_out_of_bounds() {
            let mut heap = RegionHeap::default();
            let tuple = heap.tuple_from_slice(&[]).unwrap();

            assert_eq!(
                tuple.get_element(OneBasedIndex::new(2).unwrap()),
                Err(IndexError::OutOfBounds { index: 1, len: 0 })
            );
        }

        #[test]
        fn tuple_zerobased_index_in_bounds() {
            let mut heap = RegionHeap::default();
            let tuple = heap.tuple_from_slice(&[fixnum!(0)]).unwrap();

            assert_eq!(tuple.get_element(ZeroBasedIndex::default()), Ok(fixnum!(0)));
        }

        #[test]
        fn tuple_onebased_index_in_bounds() {
            let mut heap = RegionHeap::default();
            let tuple = heap.tuple_from_slice(&[fixnum!(0)]).unwrap();

            assert_eq!(tuple.get_element(OneBasedIndex::default()), Ok(fixnum!(0)));
        }
    }

    mod set_element {
        use super::*;

        #[test]
        fn tuple_zerobased_index_out_of_bounds() {
            let mut heap = RegionHeap::default();
            let mut tuple = heap.tuple_from_slice(&[]).unwrap();
            let index = ZeroBasedIndex::new(1);

            assert_eq!(
                tuple.set_element(index, Term::NIL),
                Err(IndexError::OutOfBounds { index: 1, len: 0 })
            );
        }

        #[test]
        fn tuple_onebased_index_out_of_bounds() {
            let mut heap = RegionHeap::default();
            let mut tuple = heap.tuple_from_slice(&[]).unwrap();
            let index = OneBasedIndex::new(2).unwrap();

            assert_eq!(
                tuple.set_element(index, Term::NIL),
                Err(IndexError::OutOfBounds { index: 1, len: 0 })
            );
        }

        #[test]
        fn tuple_zerobased_index_in_bounds() {
            let mut heap = RegionHeap::default();
            let mut tuple = heap.tuple_from_slice(&[fixnum!(0)]).unwrap();
            let index = ZeroBasedIndex::default();

            assert_eq!(tuple.set_element(index, fixnum!(1234)), Ok(()),);
            assert_eq!(tuple.get_element(index), Ok(fixnum!(1234)))
        }

        #[test]
        fn tuple_onebased_index_in_bounds() {
            let mut heap = RegionHeap::default();
            let mut tuple = heap.tuple_from_slice(&[fixnum!(0)]).unwrap();
            let index = OneBasedIndex::default();

            assert_eq!(tuple.set_element(index, fixnum!(1234)), Ok(()),);
            assert_eq!(tuple.get_element(index), Ok(fixnum!(1234)))
        }
    }

    mod eq {
        use super::*;

        #[test]
        fn tuple_without_element() {
            let mut heap = RegionHeap::default();
            let lhs = heap.tuple_from_slice(&[]).unwrap();
            let rhs = heap.tuple_from_slice(&[]).unwrap();

            assert_eq!(lhs, lhs);
            assert_eq!(lhs, rhs);
            assert_eq!(rhs, lhs);
        }

        #[test]
        fn tuple_with_unequal_length() {
            let mut heap = RegionHeap::default();
            let lhs = heap.tuple_from_slice(&[fixnum!(0)]).unwrap();
            let rhs = heap.tuple_from_slice(&[fixnum!(0), fixnum!(1)]).unwrap();

            assert_ne!(lhs, rhs);
            assert_ne!(rhs, lhs);
        }
    }

    mod iter {
        use super::*;

        #[test]
        fn tuple_without_elements() {
            let mut heap = RegionHeap::default();
            let tuple = heap.tuple_from_slice(&[]).unwrap();

            assert_eq!(tuple.iter().count(), 0);
            assert_eq!(tuple.len(), 0);
        }

        #[test]
        fn tuple_with_elements() {
            let align = core::mem::align_of::<usize>();
            let layout = unsafe { Layout::from_size_align_unchecked(8 * 1024, align) };
            let mut heap = RegionHeap::new(layout);
            let arc_node = Arc::new(Node::new(
                1,
                Atom::try_from_str("node@external").unwrap(),
                0,
            ));
            // one of every type
            let slice = &[
                // small integer
                fixnum!(0),
                // big integer
                heap.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
                heap.reference(scheduler::id::next(), 0)
                    .map(|r| r.into())
                    .unwrap(),
                closure(&mut heap),
                heap.float(0.0).map(|f| f.into()).unwrap(),
                heap.external_pid(arc_node, 0, 0).map(|p| p.into()).unwrap(),
                Term::NIL,
                Pid::make_term(0, 0).unwrap(),
                atom!("atom"),
                heap.tuple_from_slice(&[atom!("tuple")])
                    .unwrap()
                    .encode()
                    .unwrap(),
                heap.map_from_slice(&[(atom!("key"), atom!("value"))])
                    .unwrap()
                    .encode()
                    .unwrap(),
                heap.list_from_slice(&[atom!("list")])
                    .unwrap()
                    .unwrap()
                    .encode()
                    .unwrap(),
            ];
            let num_terms = slice.len();
            let tuple = heap.tuple_from_slice(slice).unwrap();

            assert_eq!(tuple.iter().count(), num_terms);
            assert_eq!(tuple.len(), num_terms);
        }
    }

    mod len {
        use super::*;

        #[test]
        fn tuple_without_elements() {
            let mut heap = RegionHeap::default();
            let tuple = Tuple::new(&mut heap, 0).unwrap();

            assert_eq!(tuple.len(), 0);
        }

        #[test]
        fn tuple_with_elements() {
            let mut heap = RegionHeap::default();
            let tuple = Tuple::new(&mut heap, 3).unwrap();

            assert_eq!(tuple.len(), 3);
        }
    }

    fn closure<H: TermAlloc>(heap: &mut H) -> Term {
        let module = Atom::try_from_str("module").unwrap();
        let function = Atom::try_from_str("function").unwrap();
        let arity = 0;

        extern "C" fn native() -> Term {
            Term::NONE
        }

        heap.export_closure(module, function, arity, Some(native as *const c_void))
            .unwrap()
            .into()
    }
}

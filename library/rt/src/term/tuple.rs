use alloc::alloc::{AllocError, Allocator, Layout};
use core::convert::AsRef;
use core::fmt::{self, Debug};
use core::hash::{Hash, Hasher};
use core::ops::{Deref, Range};

use firefly_alloc::heap::Heap;

use crate::cmp::ExactEq;
use crate::gc::Gc;

use super::{Boxable, Header, LayoutBuilder, OpaqueTerm, Tag, Term, TupleIndex};

#[repr(C)]
pub struct Tuple {
    header: Header,
    elements: [OpaqueTerm],
}
impl Tuple {
    /// Creates a new tuple in the given allocator, with room for `capacity` elements
    ///
    /// # Safety
    ///
    /// It is not safe to use the pointer returned from this function without first initializing all
    /// of the tuple elements with valid values. This function does not guarantee that the elements are
    /// in any particular state, so use of the tuple without the initialization step is undefined behavior.
    pub fn new_in<A: ?Sized + Allocator>(
        capacity: usize,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        let mut this = Gc::<Tuple>::with_capacity_zeroed_in(capacity, alloc)?;
        this.header = Header::new(Tag::Tuple, capacity);
        Ok(this)
    }

    /// Creates a new tuple in the given allocator, from a slice of elements.
    ///
    /// This is a safer alternative to `new_in`, and ensures that the resulting pointer is valid for use
    /// right away.
    pub fn from_slice<A: ?Sized + Allocator>(
        slice: &[OpaqueTerm],
        alloc: &A,
    ) -> Result<Gc<Tuple>, AllocError> {
        let mut this = Self::new_in(slice.len(), alloc)?;
        this.copy_from_slice(slice);
        Ok(this)
    }

    /// Gets the size of this tuple
    #[inline]
    pub fn len(&self) -> usize {
        self.header.arity()
    }

    /// Returns true if this tuple has no elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the element at 0-based index `index`
    ///
    /// If the index is out of bounds, returns `None`
    #[inline]
    pub fn get(&self, index: usize) -> Option<OpaqueTerm> {
        self.elements.get(index).copied()
    }

    /// Returns the element at the given 0-based index without bounds checks
    ///
    /// # Safety
    ///
    /// Calling this function with an out-of-bounds index is undefined behavior
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> OpaqueTerm {
        *self.elements.get_unchecked(index)
    }

    /// Like `get` but with either 0 or 1-based indexing.
    ///
    /// Returns `None` if the index is out of bounds
    #[inline]
    pub fn get_element<I: TupleIndex>(&self, index: I) -> Option<OpaqueTerm> {
        self.get(index.into())
    }

    /// Produces a new tuple with the element at `index` set to `value`
    ///
    /// Consider using `set_element_mut` if the caller can guarantee that no
    /// other references to this tuple exist.
    ///
    /// This function will panic if the index is out of bounds
    pub fn set_element<A: ?Sized + Allocator, I: TupleIndex, V: Into<OpaqueTerm>>(
        &self,
        index: I,
        value: V,
        alloc: &A,
    ) -> Result<Gc<Tuple>, AllocError> {
        let index: usize = index.into();
        if index >= self.len() {
            panic!(
                "invalid index {}, exceeds max length of {}",
                index,
                self.len()
            );
        }

        let mut tuple = Self::new_in(self.len(), alloc)?;
        tuple.copy_from_slice(self.as_slice());

        unsafe {
            *tuple.elements.get_unchecked_mut(index) = value.into();
        }

        Ok(tuple)
    }

    /// Mutates this tuple in place, setting the element at `index` to `value`
    ///
    /// This function will panic if the index is out of bounds
    pub fn set_element_mut<I: TupleIndex, V: Into<OpaqueTerm>>(&mut self, index: I, value: V) {
        let index: usize = index.into();
        let element = self
            .elements
            .get_mut(index)
            .expect("invalid tuple index, out of bounds");
        *element = value.into();
    }

    /// Copies all of the elements from `slice` into this tuple
    ///
    /// NOTE: The slice and this tuple are asserted to be the same length
    pub fn copy_from_slice(&mut self, slice: &[OpaqueTerm]) {
        assert_eq!(self.elements.len(), slice.len());
        self.elements.copy_from_slice(slice)
    }

    /// Get this tuple as a slice of raw elements
    #[inline]
    pub fn as_slice(&self) -> &[OpaqueTerm] {
        &self.elements
    }

    /// Get this tuple as a mutable slice of raw elements
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [OpaqueTerm] {
        &mut self.elements
    }

    /// Get an iterator over the elements of this tuple as `Term`
    pub fn iter(&self) -> TupleIter<'_> {
        TupleIter::new(self)
    }
}
impl Boxable for Tuple {
    type Metadata = usize;

    const TAG: Tag = Tag::Tuple;

    #[inline]
    fn header(&self) -> &Header {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn layout_excluding_heap<H: ?Sized + Heap>(&self, heap: &H) -> Layout {
        if heap.contains((self as *const Self).cast()) {
            return Layout::new::<()>();
        }

        let mut builder = LayoutBuilder::new();
        for element in self.elements.iter().copied() {
            if element.is_gcbox() || element.is_nonempty_list() || element.is_tuple() {
                let element: Term = element.into();
                builder.extend(&element);
            }
        }
        builder += Layout::for_value(self);
        builder.finish()
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let mut cloned = Self::new_in(self.len(), heap).unwrap();
            let elements = cloned.as_mut_slice();
            for (i, element) in self.elements.iter().copied().enumerate() {
                if element.is_gcbox() || element.is_nonempty_list() || element.is_tuple() {
                    let element: Term = element.into();
                    unsafe {
                        *elements.get_unchecked_mut(i) = element.unsafe_clone_to_heap(heap).into();
                    }
                } else {
                    element.maybe_increment_refcount();
                    unsafe {
                        *elements.get_unchecked_mut(i) = element;
                    }
                }
            }
            cloned
        }
    }
}
impl Tuple {
    pub unsafe fn unsafe_move_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        use crate::term::Cons;

        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            Gc::from_raw(ptr.cast_mut())
        } else {
            let mut cloned = Self::new_in(self.len(), heap).unwrap();
            let elements = cloned.as_mut_slice();
            for (i, element) in self.elements.iter().copied().enumerate() {
                if element.is_rc() {
                    *elements.get_unchecked_mut(i) = element;
                    continue;
                }
                if element.is_nonempty_list() {
                    let mut list = Gc::from_raw(element.as_ptr() as *mut Cons);
                    let moved = list.unsafe_move_to_heap(heap);
                    *elements.get_unchecked_mut(i) = moved.into();
                    continue;
                }
                if element.is_gcbox() || element.is_tuple() {
                    let term: Term = element.into();
                    let moved = term.unsafe_move_to_heap(heap);
                    *elements.get_unchecked_mut(i) = moved.into();
                    continue;
                }
                *elements.get_unchecked_mut(i) = element;
            }
            cloned
        }
    }
}
impl AsRef<[OpaqueTerm]> for Tuple {
    fn as_ref(&self) -> &[OpaqueTerm] {
        &self.elements
    }
}
impl Debug for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("{")?;
        for (i, element) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", {:?}", element)?;
            } else {
                write!(f, "{:?}", element)?;
            }
        }
        f.write_str("}")
    }
}
impl fmt::Display for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("{")?;
        for (i, element) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", {}", element)?;
            } else {
                write!(f, "{}", element)?;
            }
        }
        f.write_str("}")
    }
}
impl PartialOrd for Tuple {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Tuple {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        let by_len = self.len().cmp(&other.len());
        if by_len != Ordering::Equal {
            return by_len;
        }

        for (x, y) in self.iter().zip(other.iter()) {
            let result = x.cmp(&y);
            match result {
                Ordering::Less | Ordering::Greater => return result,
                _ => continue,
            }
        }

        Ordering::Equal
    }
}
impl Hash for Tuple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for item in self.iter() {
            item.hash(state);
        }
    }
}

impl Eq for Tuple {}
impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}
impl PartialEq<Gc<Tuple>> for Tuple {
    fn eq(&self, other: &Gc<Tuple>) -> bool {
        self.eq(other.deref())
    }
}
impl ExactEq for Tuple {
    fn exact_eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(x, y)| x.exact_eq(&y))
    }
}

pub struct TupleIter<'a> {
    tuple: &'a Tuple,
    alive: Range<usize>,
}
impl TupleIter<'_> {
    pub fn new<'a>(tuple: &'a Tuple) -> TupleIter<'a> {
        let alive = match tuple.len() {
            0 => 0..0,
            n => 0..n,
        };
        TupleIter { tuple, alive }
    }
}
impl<'a> Iterator for TupleIter<'a> {
    type Item = Term;

    fn next(&mut self) -> Option<Self::Item> {
        self.alive
            .next()
            .map(|idx| unsafe { self.tuple.get_unchecked(idx).into() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}
impl<'a> core::iter::DoubleEndedIterator for TupleIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.alive
            .next_back()
            .map(|idx| unsafe { self.tuple.get_unchecked(idx).into() })
    }
}
impl<'a> core::iter::ExactSizeIterator for TupleIter<'a> {
    fn len(&self) -> usize {
        self.alive.end - self.alive.start
    }

    fn is_empty(&self) -> bool {
        self.alive.is_empty()
    }
}
impl<'a> core::iter::FusedIterator for TupleIter<'a> {}
unsafe impl<'a> core::iter::TrustedLen for TupleIter<'a> {}

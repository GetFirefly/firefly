use alloc::alloc::{AllocError, Allocator, Layout};
use core::any::TypeId;
use core::convert::AsRef;
use core::fmt::{self, Debug};
use core::hash::{Hash, Hasher};
use core::ptr::{self, NonNull};

use super::{OpaqueTerm, Term, Type, Value};

pub struct Tuple([OpaqueTerm]);
impl Tuple {
    pub const TYPE_ID: TypeId = TypeId::of::<Tuple>();

    /// Creates a new tuple in the given allocator, with room for `capacity` elements
    ///
    /// # Safety
    ///
    /// It is not safe to use the pointer returned from this function without first initializing all
    /// of the tuple elements with valid values. This function does not guarantee that the elements are
    /// in any particular state, so use of the tuple without the initialization step is undefined behavior.
    pub unsafe fn new_in<A: Allocator>(
        capacity: usize,
        alloc: A,
    ) -> Result<NonNull<Tuple>, AllocError> {
        let (layout, value_offset) = Layout::<usize>::new()
            .extend(Layout::<OpaqueTerm>::array(capacity).unwrap())
            .unwrap();
        let ptr: NonNull<u8> = alloc.allocate(layout)?.cast();
        let ptr = ptr::from_raw_parts_mut(ptr.add(value_offset).cast(), capacity);
        Ok(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Creates a new tuple in the given allocator, from a slice of elements.
    ///
    /// This is a safer alternative to `new_in`, and ensures that the resulting pointer is valid for use
    /// right away.
    pub fn from_slice<A: Allocator>(
        slice: &[OpaqueTerm],
        alloc: A,
    ) -> Result<NonNull<Tuple>, AllocError> {
        let tuple = Self::new_in(slice.len(), alloc)?;
        tuple.as_mut().copy_from_slice(slice);
        tuple
    }

    /// Gets the size of this tuple
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if this tuple has no elements
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the element at 0-based index `index` as a `Value`
    ///
    /// If the index is out of bounds, returns `None`
    pub fn get(&self, index: usize) -> Option<Value> {
        self.0.get(index).copied().map(|term| term.into())
    }

    /// Returns the element at the given 0-based index as a `Value` without bounds checks
    ///
    /// # Safety
    ///
    /// Calling this function with an out-of-bounds index is undefined behavior
    pub unsafe fn get_unchecked(&self, index: usize) -> Value {
        let term = *self.0.get_unchecked(index);
        term.into()
    }

    /// Like `get` but with either 0 or 1-based indexing.
    #[inline]
    pub fn get_element<I: TupleIndex>(&self, index: I) -> anyhow::Result<Value> {
        let index: usize = index.into();
        self.get(index).ok_or_else(|| {
            anyhow!(
                "invalid index {}, exceeds max length of {}",
                index,
                self.len()
            )
        })
    }

    /// Sets the element at the given index
    #[inline]
    pub fn set_element<I: TupleIndex, V: Into<OpaqueTerm>>(
        &mut self,
        index: I,
        value: V,
    ) -> anyhow::Result<()> {
        let index: usize = index.into();
        if let Some(element) = self.0.get_mut(index) {
            *element = value.into();
            return Ok(());
        }

        let len = self.len();
        Err(anyhow!(
            "invalid index {}, exceeds max length of {}",
            index,
            len
        ))
    }

    /// Copies all of the elements from `slice` into this tuple
    ///
    /// NOTE: The slice and this tuple are asserted to be the same length
    pub fn copy_from_slice(&mut self, slice: &[OpaqueTerm]) {
        assert_eq!(self.len(), slice.len());
        self.0.copy_from_slice(slice)
    }

    /// Get this tuple as a slice of raw elements
    #[inline]
    pub fn as_slice(&self) -> &[OpaqueTerm] {
        &self.0
    }

    /// Get an iterator over the elements of this tuple as `Value`
    pub fn iter(&self) -> TupleIter<'_> {
        TupleIter::new(self)
    }
}
impl AsRef<[OpaqueTerm]> for Tuple {
    fn as_ref(&self) -> &[OpaqueTerm] {
        self.0.as_ref()
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
impl PartialOrd for Tuple {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Tuple {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
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
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.alive
            .next()
            .map(|idx| unsafe { self.tuple.get_unchecked(idx) })
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
            .map(|idx| unsafe { self.tuple.get_unchecked(idx) })
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
impl<'a> core::iter::TrustedLen for TupleIter<'a> {}

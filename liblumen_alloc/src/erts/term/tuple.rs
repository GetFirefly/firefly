use core::alloc::Layout;
use core::cmp;
use core::convert::TryInto;
use core::fmt;
use core::iter::FusedIterator;
use core::mem;
use core::ptr;

use crate::borrow::CloneToProcess;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexError {
    OutOfBounds { len: usize, index: usize },
    BadArgument(Term),
}
impl IndexError {
    pub fn new(index: usize, len: usize) -> Self {
        Self::OutOfBounds { len, index }
    }
}
impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Self::OutOfBounds { len, index } => {
                write!(f, "invalid index {}: exceeds max length of {}", index, len)
            }
            &Self::BadArgument(term) => write!(f, "invalid index: bad argument {:?}", term),
        }
    }
}
impl From<BadArgument> for IndexError {
    fn from(badarg: BadArgument) -> Self {
        Self::BadArgument(badarg.argument())
    }
}

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
pub struct Tuple {
    header: Term,
}
impl Tuple {
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

    /// Returns the size of this tuple
    #[inline]
    pub fn size(&self) -> usize {
        self.header.arityval()
    }

    /// Returns a pointer to the head element
    ///
    /// NOTE: This is unsafe to use unless you know the tuple has been allocated
    #[inline]
    pub fn head(&self) -> *mut Term {
        unsafe { (self as *const Tuple).offset(1) as *mut Term }
    }

    /// This function produces a `Layout` which represents the memory layout
    /// needed for the tuple header, and `num_elements` terms. The resulting
    /// size is only enough for the tuple and word-sized values, e.g. immediates
    /// or boxes. You need to extend this layout with others representing more
    /// complex values like maps/lists/etc., if you want a layout that covers all
    /// the memory needed by elements of the tuple
    #[inline]
    pub fn layout(num_elements: usize) -> Layout {
        let size = mem::size_of::<Self>() + (num_elements * mem::size_of::<Term>());
        unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<Term>()) }
    }

    /// Constructs an iterator over elements of the tuple
    #[inline]
    pub fn iter(&self) -> TupleIter {
        TupleIter::new(self)
    }

    /// Sets an element in the tuple, returns `Ok(())` if successful,
    /// otherwise returns `Err(IndexErr)` if the given index is invalid
    #[inline]
    pub fn set_element(&mut self, index: Term, element: Term) -> Result<(), IndexError> {
        let size = self.size();
        if let Ok(TypedTerm::SmallInteger(small)) = index.to_typed_term() {
            match small.try_into() {
                Ok(i) if i > 0 && i <= size => Ok(self.do_set_element(i, element)),
                Ok(i) => Err(IndexError::new(i, size)),
                Err(_) => Err(BadArgument::new(index).into()),
            }
        } else {
            Err(BadArgument::new(index).into())
        }
    }

    /// Like `set_element` but for internal runtime use, as it takes a `usize` directly
    #[inline]
    pub fn set_element_internal(&mut self, index: usize, element: Term) -> Result<(), IndexError> {
        let size = self.size();
        if index > 0 && index <= size {
            Ok(self.do_set_element(index, element))
        } else {
            Err(IndexError::new(index, size))
        }
    }

    #[inline]
    fn do_set_element(&mut self, index: usize, element: Term) {
        assert!(index > 0 && index <= self.size());
        assert!(element.is_boxed() || element.is_list() || element.is_immediate());
        unsafe {
            let ptr = self.head().offset((index - 1) as isize);
            ptr::write(ptr, element);
        }
    }

    /// Fetches an element from the tuple, returns `Ok(term)` if the index is
    /// valid, otherwise returns `Err(IndexErr)`
    #[inline]
    pub fn get_element(&self, index: Term) -> Result<Term, IndexError> {
        let size = self.size();
        if let Ok(TypedTerm::SmallInteger(small)) = index.to_typed_term() {
            match small.try_into() {
                Ok(i) if i > 0 && i <= size => Ok(self.do_get_element(i)),
                Ok(i) => Err(IndexError::new(i, size)),
                Err(_) => Err(BadArgument::new(index).into()),
            }
        } else {
            Err(BadArgument::new(index).into())
        }
    }

    /// Like `get_element` but for internal runtime use, as it takes a `usize`
    /// directly, rather than a value of type `Term`
    #[inline]
    pub fn get_element_internal(&self, index: usize) -> Result<Term, IndexError> {
        let size = self.size();
        if index > 0 && index <= size {
            Ok(self.do_get_element(index))
        } else {
            Err(IndexError::new(index, size))
        }
    }

    #[inline]
    fn do_get_element(&self, index: usize) -> Term {
        assert!(index > 0 && index <= self.size());
        unsafe {
            let ptr = self.head().offset((index - 1) as isize);
            follow_moved(*ptr)
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
    fn clone_to_process<A: AllocInProcess>(&self, process: &mut A) -> Term {
        // The result of calling this will be a Tuple with everything located
        // contigously in memory
        unsafe {
            // Allocate the space needed for the header and all the elements
            let num_elements = self.size();
            let size = mem::size_of::<Self>() + (num_elements * mem::size_of::<Term>());
            let layout = Layout::from_size_align_unchecked(size, mem::align_of::<Term>());
            let ptr = process.alloc_layout(layout).unwrap().as_ptr() as *mut Self;
            // Get pointer to the old head element location
            let old_head = self.head();
            // Get pointer to the new head element location
            let head = ptr.offset(1) as *mut Term;
            // Write the header
            ptr::write(
                ptr,
                Self {
                    header: self.header,
                },
            );
            // Write each element
            for offset in 0..num_elements {
                let old = *old_head.offset(offset as isize);
                if old.is_immediate() {
                    ptr::write(head.offset(offset as isize), old);
                } else {
                    // Recursively call clone_to_process, and then write the box header here
                    let boxed = old.clone_to_process(process);
                    ptr::write(head.offset(offset as isize), boxed);
                }
            }
            let tuple = &*ptr;
            tuple.as_term()
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

        match self.size().cmp(&other.size()) {
            Ordering::Less => return Some(Ordering::Less),
            Ordering::Greater => return Some(Ordering::Greater),
            Ordering::Equal => self.iter().partial_cmp(other.iter()),
        }
    }
}
impl fmt::Debug for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Tuple(")?;
        for element in self.iter() {
            f.write_fmt(format_args!("{:?}", element.as_usize()))?;
        }
        f.write_str(")")
    }
}

pub struct TupleIter {
    head: *mut Term,
    pos: usize,
    last_pos: usize,
}
impl TupleIter {
    pub fn new(tuple: &Tuple) -> Self {
        match tuple.size() {
            0 => Self { head: tuple.head(), pos: 0, last_pos: 0 },
            n => Self { head: tuple.head(), pos: 0, last_pos: n - 1 },
        }
    }
}
impl Iterator for TupleIter {
    type Item = Term;

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.last_pos == 0 {
            (0, Some(0))
        } else {
            (self.last_pos + 1, Some(self.last_pos + 1))
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos > self.last_pos {
            return None;
        }
        if self.last_pos == 0 {
            // This occurs with empty tuples only
            return None;
        }
        let term = unsafe { self.head.offset(self.pos as isize) };
        self.pos += 1;
        unsafe { Some(*term) }
    }
}
impl FusedIterator for TupleIter {}
impl ExactSizeIterator for TupleIter {
    fn len(&self) -> usize {
        self.last_pos + 1
    }

    fn is_empty(&self) -> bool {
        self.last_pos == 0
    }
}

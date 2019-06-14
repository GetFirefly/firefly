use core::cmp;
use core::ptr;
use core::mem;
use core::iter::FusedIterator;
use core::alloc::Layout;

use crate::borrow::CloneToProcess;
use crate::erts::ProcessControlBlock;

use super::{AsTerm, Term};

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
#[derive(Debug)]
pub struct Tuple {
    head: *mut Term,
    size: usize,
}
impl Tuple {
    pub unsafe fn from_raw_parts(head: *mut Term, size: usize) -> Self {
        Self { head, size }
    }

    pub fn iter(&self) -> TupleIter {
        TupleIter::new(self)
    }
}
unsafe impl AsTerm for Tuple {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::from_raw(&self.size as *const _ as usize)
    }
}
impl CloneToProcess for Tuple {
    fn clone_to_process(&self, process: &mut ProcessControlBlock) -> Term {
        // The result of calling this will be a Tuple with everything located
        // contigously in memory
        unsafe {
            // Allocate the space needed for the header and all the elements
            let size = mem::size_of::<Self>() + self.size;
            let layout = Layout::from_size_align_unchecked(size, mem::align_of::<Term>());
            let ptr = process.alloc_layout(layout).unwrap().as_ptr() as *mut Self;
            // Get pointer to the new head element location
            let head = ptr.offset(1) as *mut Term;
            // Write the header
            ptr::write(ptr, Self { head, size: self.size });
            // Write each element
            for offset in 0..self.size {
                let old = *self.head.offset(offset as isize);
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

        match self.size.cmp(&other.size) {
            Ordering::Less => return Some(Ordering::Less),
            Ordering::Greater => return Some(Ordering::Greater),
            Ordering::Equal => self.iter().partial_cmp(other.iter()),
        }
    }
}

pub struct TupleIter {
    head: *mut Term,
    size: usize,
    pos: usize,
}
impl TupleIter {
    pub fn new(tuple: &Tuple) -> Self {
        Self {
            head: tuple.head,
            size: tuple.size,
            pos: 0,
        }
    }
}
impl Iterator for TupleIter {
    type Item = Term;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.size - 1 {
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
        self.size
    }

    fn is_empty(&self) -> bool {
        self.size == 0
    }
}

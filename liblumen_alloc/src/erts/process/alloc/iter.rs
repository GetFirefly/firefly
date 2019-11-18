use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem;

use crate::erts;
use crate::erts::term::prelude::{Closure, Encoded, HeapBin, Term};

use super::Heap;

pub trait HeapIter: Sized + Heap {
    /// This function produces an iterator over terms on the heap that form
    /// nodes in the reference tree (i.e. boxed/list pointers, and headers).
    /// Each value produced by the iterator is a mutable reference to the term
    /// on the heap.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for term in self.iter_mut() {
    ///   if term.is_boxed() {
    ///       ...
    ///   }
    /// }
    fn iter_mut<'a>(&mut self) -> IterMut<'a, Self>;
}

/// This implementation relies on the fact that access to the heap is exclusive,
/// and that it is pinned in memory for the duration of the iterator, in order to
/// intentionally, but safely, break the rules on the construction of mutable
/// references to allow modifying the heap while iterating over it.
///
/// We know for our purposes that this is safe because our iterator takes into account
/// the fact that the pointer to the top of the heap may shift, so any allocations
/// that take place during iteration will be picked up by the iterator.
impl<T> HeapIter for T
where
    T: Heap,
{
    fn iter_mut<'a>(&mut self) -> IterMut<'a, Self> {
        IterMut {
            heap: self as *const _ as *mut Self,
            pos: self.heap_start(),
            _marker: PhantomData,
        }
    }
}

pub struct IterMut<'a, T: Heap> {
    heap: *mut T,
    pos: *mut Term,
    _marker: PhantomData<&'a mut Term>,
}
unsafe impl<T: Heap + Sync> Sync for IterMut<'_, T> {}
unsafe impl<T: Heap + Send> Send for IterMut<'_, T> {}
impl<'a, T> Iterator for IterMut<'a, T>
where
    T: Heap,
{
    type Item = &'a mut Term;

    fn next(&mut self) -> Option<Self::Item> {
        use crate::erts::term::prelude::UnsizedBoxable;

        let heap = unsafe { &*self.heap };
        // Seek to the next valid term (boxed/list pointer or header)
        // We skip immediates because we only care about terms that
        // form roots/branches in the reference tree - all immediates
        // are either on the stack, or are leaves in the reference tree,
        // and so can be skipped
        loop {
            let pos = self.pos;
            // Stop when we catch up to heap_top
            if pos < heap.heap_top() {
                // Get reference to term
                let term = unsafe { &mut *pos };
                // If we have a valid term, return the reference, otherwise try again
                if term.is_boxed() || term.is_non_empty_list() {
                    // Boxes are word-sized
                    self.pos = unsafe { pos.add(1) };
                    return Some(term);
                } else if term.is_header() {
                    // For certain terms, we need to walk their elements
                    if term.is_tuple() {
                        // Tuple header is word-sized, followed by elements
                        self.pos = unsafe { pos.add(1) };
                        // Shift to first element
                        return Some(term);
                    } else if term.is_function() {
                        let closure_box = unsafe { Closure::from_raw_term(pos) };
                        let closure = closure_box.as_ref();
                        // When there is env to check, set position to beginning of
                        // environment so that we can walk each item in the environment
                        //
                        // Otherwise, skip to the next term
                        if closure.env_len() > 0 {
                            let env_pos = closure.env_slice() as *const _ as *mut Term;
                            self.pos = env_pos;
                        } else {
                            let arity = erts::to_word_size(mem::size_of_val(closure));
                            self.pos = unsafe { pos.add(arity) };
                        }
                        return Some(term);
                    } else if term.is_heapbin() {
                        // Like closures, heap binaries are dynamically sized
                        let bin_box = unsafe { HeapBin::from_raw_term(pos) };
                        let bin = bin_box.as_ref();
                        let arity = erts::to_word_size(mem::size_of_val(bin));
                        self.pos = unsafe { pos.add(arity) };

                        return Some(term);
                    } else {
                        // Calculate size of term to find next term address
                        let arity = term.arity() + 1;
                        // Shift position to beginning of next term
                        self.pos = unsafe { pos.add(arity) };
                        return Some(term);
                    }
                } else {
                    self.pos = unsafe { pos.add(1) };
                    continue;
                }
            } else {
                // We're at the end of the heap
                return None;
            }
        }
    }
}
// We always return None when positioned at the end of the heap,
// so we can return None arbitrarily many times if needed
impl<T: Heap> FusedIterator for IterMut<'_, T> {}

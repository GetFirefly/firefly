use core::mem;
use core::ptr;

use crate::erts::*;

use liblumen_core::util::pointer::{distance_absolute, in_area};

/// This type represents the old generation process heap
///
/// This heap has no stack, and is only swept when new values are tenured
pub struct OldHeap {
    pub(in crate::erts::process) start: *mut Term,
    pub(in crate::erts::process) end: *mut Term,
    pub(in crate::erts::process) top: *mut Term,
}
impl OldHeap {
    /// Returns a new instance which manages the memory represented
    /// by `start -> start + size`. If `start` is the null pointer,
    /// then this is considered an empty, inactive heap, and will
    /// return sane values for all functions, but will not participate
    /// in collections
    #[inline]
    pub fn new(start: *mut Term, size: usize) -> Self {
        if start.is_null() {
            Self::empty()
        } else {
            let end = unsafe { start.offset(size as isize) };
            let top = start;
            Self { start, end, top }
        }
    }

    /// Returns an empty, inactive default instance which can be
    /// activated by passing `reset` the same arguments as `new`
    #[inline]
    pub fn empty() -> Self {
        Self {
            start: ptr::null_mut(),
            end: ptr::null_mut(),
            top: ptr::null_mut(),
        }
    }

    /// Returns true if this heap has been allocated memory,
    /// otherwise returns false. Being inactive implies that
    /// the owning process has not yet undergone tenuring of
    /// objects, or it just completed a full sweep
    #[inline]
    pub fn active(&self) -> bool {
        !self.start.is_null()
    }

    /// Returns the total allocated size of this heap
    #[inline]
    pub fn size(&self) -> usize {
        distance_absolute(self.end, self.start)
    }

    /// Returns the used size of this heap
    #[inline]
    pub fn heap_used(&self) -> usize {
        distance_absolute(self.top, self.start)
    }

    /// Returns the space remaining in this heap
    #[inline]
    pub fn heap_available(&self) -> usize {
        distance_absolute(self.end, self.top)
    }

    /// Returns true if the given pointer points into this heap
    #[inline]
    pub fn contains<T>(&self, term: *const T) -> bool {
        in_area(term, self.start, self.top)
    }

    /// This function walks the heap, applying the given function to every term
    /// encountered that is either boxed, a list, or a header value. All other
    /// term values are skipped over as they are components of one of the above
    /// mentioned types.
    ///
    /// The callback provided will receive the current `Term` and pointer to that
    /// term (also functions as the current position in the heap). This can be used
    /// to read/write data to/from the heap at that position.
    ///
    /// The callback should return `Some(pos)` if traversal should continue,
    /// returning the pointer `pos` as the position at which to resume traversing the
    /// heap. This pointer must _always_ be incremented by at least `1` to ensure that
    /// an infinite loop does not occur, and this will be verified when run in debug mode.
    /// If `None` is returned, traversal will stop.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// self.walk(|heap, term, pos| {
    ///     if term.is_tuple() {
    ///         let arity = term.arityval();
    ///         # ...
    ///         return Some(unsafe { pos.offset(arity + 1) });
    ///     }
    ///     None
    /// });
    #[inline]
    pub fn walk<F>(&mut self, mut fun: F)
    where
        F: FnMut(&mut OldHeap, Term, *mut Term) -> Option<*mut Term>,
    {
        let mut pos = self.start;

        while (pos as usize) < (self.top as usize) {
            let prev_pos = pos;
            let term = unsafe { *pos };
            if term.is_boxed() {
                if let Some(new_pos) = fun(self, term, pos) {
                    pos = new_pos;
                } else {
                    break;
                }
            } else if term.is_list() {
                if let Some(new_pos) = fun(self, term, pos) {
                    pos = new_pos;
                } else {
                    break;
                }
            } else if term.is_header() {
                if let Some(new_pos) = fun(self, term, pos) {
                    pos = new_pos;
                } else {
                    break;
                }
            } else {
                pos = unsafe { pos.offset(1) };
            }

            debug_assert!(
                pos as usize > prev_pos as usize,
                "walk callback must advance position in heap by at least 1!"
            );
        }
    }

    /// Moves a boxed term into this heap
    ///
    /// - `orig` is the original boxed term pointer
    /// - `ptr` is the pointer contained in the box
    /// - `header` is the header value pointed to by `ptr`
    ///
    /// When this function returns, `orig` will have been updated
    /// with a new boxed term which points into this heap rather than
    /// the previous location. Likewise, `ptr` will have been updated
    /// with the same value as `orig`, forwarding references to this heap.
    #[inline]
    pub unsafe fn move_into(&mut self, orig: *mut Term, ptr: *mut Term, header: Term) -> usize {
        assert!(header.is_header());

        // All other headers follow more or less the same formula,
        // the header arityval contains the size of the term in words,
        // so we simply need to move those words into this heap
        let mut heap_top = self.top;

        // Sub-binaries are a little different, in that since we're garbage
        // collecting, we can't guarantee that the original binary will stick
        // around, and we don't necessarily want it to. Instead we create a new
        // heap binary (if within the size limit) that contains the slice of
        // the original binary that the sub-binary referenced. If the sub-binary
        // is too large to build a heap binary, then the original must be a ProcBin,
        // so we don't need to worry about it being collected out from under us
        // TODO: Handle other subtype cases, see erl_gc.h:60
        if header.is_subbinary() {
            let bin = &*(ptr as *mut SubBinary);
            // Convert to HeapBin if applicable
            if let Ok((bin_header, bin_flags, bin_ptr, bin_size)) = bin.to_heapbin_parts() {
                // Save space for box
                let dst = heap_top.offset(1);
                // Create box pointing to new destination
                let val = Term::from_raw(dst as usize | Term::FLAG_BOXED);
                ptr::write(heap_top, val);
                let dst = dst as *mut HeapBin;
                let new_bin_ptr = dst.offset(1) as *mut u8;
                // Copy referenced part of binary to heap
                ptr::copy_nonoverlapping(bin_ptr, new_bin_ptr, bin_size);
                // Write heapbin header
                ptr::write(dst, HeapBin::from_raw_parts(bin_header, bin_flags));
                // Write a move marker to the original location
                let marker = Term::from_raw(heap_top as usize | Term::FLAG_BOXED);
                ptr::write(orig, marker);
                // Update `ptr` as well
                ptr::write(ptr, marker);
                // Update top pointer
                self.top = new_bin_ptr.offset(bin_size as isize) as *mut Term;
                // We're done
                return 1 + to_word_size(mem::size_of::<HeapBin>() + bin_size);
            }
        }

        let mut num_elements = header.arityval();
        let moved = num_elements + 1;

        // Write the a move marker to the original location
        let marker = Term::from_raw(heap_top as usize | Term::FLAG_BOXED);
        ptr::write(orig, marker);
        // And to `ptr` as well
        ptr::write(ptr, marker);
        // Move `ptr` to the first arityval
        let mut ptr = ptr.offset(1);
        // Write the term header to the location pointed to by the boxed term
        ptr::write(heap_top, header);
        // Move heap_top to the first element location
        heap_top = heap_top.offset(1);
        // For each additional term element, move to new location
        while num_elements > 0 {
            num_elements -= 1;
            ptr::write(heap_top, *ptr);
            heap_top = heap_top.offset(1);
            ptr = ptr.offset(1);
        }
        // Update the top pointer
        self.top = heap_top;
        // Return the number of words moved into this heap
        moved
    }

    /// Like `move_into`, but designed for cons cells, as the move marker approach
    /// differs slightly. The head element of the cell is set to the none value, and
    /// the tail element contains the forwarding pointer.
    #[inline]
    pub unsafe fn move_cons_into(&mut self, orig: *mut Term, ptr: *mut Cons, cons: Cons) {
        // Copy cons cell to this heap
        let location = self.top as *mut Cons;
        ptr::write(location, cons);
        // Create list term to write to original location
        let list = Term::from_raw(location as usize | Term::FLAG_LIST);
        ptr::write(orig, list);
        // Write marker to old list location
        let marker = Cons::new(Term::NONE, list);
        ptr::write(ptr as *mut Cons, marker);
        // Update the top pointer
        self.top = location.offset(1) as *mut Term;
    }

    /// This function is used during garbage collection to sweep this heap for references
    /// that reside in younger generation heaps and have not yet been moved to this heap.
    ///
    /// In more general terms, this function will walk the current heap until
    /// the top of the heap is reached, searching for any values outside this heap
    /// heap that require moving.
    pub fn sweep(&mut self) {
        self.walk(|heap: &mut Self, term: Term, pos: *mut Term| {
            unsafe {
                if term.is_boxed() {
                    let ptr = term.boxed_val();
                    let boxed = *ptr;
                    if is_move_marker(boxed) {
                        assert!(boxed.is_boxed());
                        // Overwrite move marker with "real" boxed term
                        ptr::write(pos, boxed);
                    } else if !heap.contains(ptr) {
                        // Move to top of this heap
                        heap.move_into(pos, ptr, boxed);
                    }
                    Some(pos.offset(1))
                } else if term.is_list() {
                    let ptr = term.list_val();
                    let cons = *ptr;
                    if cons.is_move_marker() {
                        // Overwrite move marker with "real" list term
                        ptr::write(pos, cons.tail);
                    } else if !heap.contains(ptr) {
                        // Move to top of this heap
                        heap.move_cons_into(pos, ptr, cons);
                    }
                    Some(pos.offset(1))
                } else if term.is_header() {
                    if term.is_tuple() {
                        // We need to check all elements, so we just skip over the tuple header
                        Some(pos.offset(1))
                    } else if term.is_match_context() {
                        let ctx = &mut *(pos as *mut MatchContext);
                        let base = ctx.base();
                        let orig = ctx.orig();
                        let orig_term = *orig;
                        let ptr = orig_term.boxed_val();
                        let bin = *ptr;
                        if is_move_marker(bin) {
                            ptr::write(orig, bin);
                            ptr::write(base, binary_bytes(bin));
                        } else if !heap.contains(ptr) {
                            heap.move_into(orig, ptr, bin);
                            ptr::write(base, binary_bytes(bin));
                        }
                        Some(pos.offset(1 + (term.arityval() as isize)))
                    } else {
                        Some(pos.offset(1 + (term.arityval() as isize)))
                    }
                } else {
                    Some(pos.offset(1))
                }
            }
        });
    }
}
impl Default for OldHeap {
    fn default() -> Self {
        Self::empty()
    }
}

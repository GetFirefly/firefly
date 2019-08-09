use core::fmt;
use core::mem;
use core::ptr;

use liblumen_core::util::pointer::*;

use crate::erts::term::binary_bytes;
use crate::erts::term::{is_move_marker, Cons, HeapBin, MatchContext, ProcBin, SubBinary};
use crate::erts::*;

use super::{VirtualBinaryHeap, YoungHeap};

/// This type represents the old generation process heap
///
/// This heap has no stack, and is only swept when new values are tenured
pub struct OldHeap {
    start: *mut Term,
    end: *mut Term,
    top: *mut Term,
    vheap: VirtualBinaryHeap,
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
            let end = unsafe { start.add(size) };
            let top = start;
            let vheap = VirtualBinaryHeap::new(size);
            Self {
                start,
                end,
                top,
                vheap,
            }
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
            vheap: VirtualBinaryHeap::new(0),
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

    /// Returns the pointer to the bottom of the heap
    #[inline(always)]
    pub fn heap_start(&self) -> *mut Term {
        self.start
    }

    /// Returns the pointer to the top of the heap
    #[inline(always)]
    pub fn heap_pointer(&self) -> *mut Term {
        self.top
    }

    /// Returns the used size of this heap
    #[inline]
    pub fn heap_used(&self) -> usize {
        distance_absolute(self.top, self.start)
    }

    #[inline]
    pub fn heap_available(&self) -> usize {
        distance_absolute(self.end, self.top)
    }

    /// Returns the used size of the virtual heap
    #[allow(unused)]
    #[inline]
    pub fn virtual_heap_used(&self) -> usize {
        self.vheap.heap_used()
    }

    /// Returns true if the given ProcBin is on this heap's virtual binary heap
    #[inline]
    pub fn virtual_heap_contains<T>(&self, term: *const T) -> bool {
        self.vheap.contains(term)
    }

    /// Unlinks the given ProcBin from the virtual binary heap, but does not free it
    #[inline]
    pub fn virtual_heap_unlink(&mut self, bin: &ProcBin) {
        self.vheap.unlink(bin)
    }

    /// Adds a binary reference to this heap's virtual binary heap
    #[inline]
    pub fn virtual_alloc(&mut self, bin: &ProcBin) -> Term {
        self.vheap.push(bin)
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
    ///         return Some(unsafe { pos.add(arity + 1) });
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
            } else if term.is_non_empty_list() {
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
                pos = unsafe { pos.add(1) };
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
        debug_assert_ne!(orig, self.top);

        // All other headers follow more or less the same formula,
        // the header arityval contains the size of the term in words,
        // so we simply need to move those words into this heap
        let heap_top = self.top;

        // Sub-binaries are a little different, in that since we're garbage
        // collecting, we can't guarantee that the original binary will stick
        // around, and we don't necessarily want it to. Instead we create a new
        // heap binary (if within the size limit) that contains the slice of
        // the original binary that the sub-binary referenced. If the sub-binary
        // is too large to build a heap binary, then the original must be a ProcBin,
        // so we don't need to worry about it being collected out from under us
        // TODO: Handle other subtype cases, see erl_gc.h:60
        if header.is_subbinary_header() {
            let bin = &*(ptr as *mut SubBinary);
            // Convert to HeapBin if applicable
            if let Ok((bin_header, bin_flags, bin_ptr, bin_size)) = bin.to_heapbin_parts() {
                // Save space for box
                let dst = heap_top.add(1);
                // Create box pointing to new destination
                let val = Term::make_boxed(dst);
                ptr::write(heap_top, val);
                let dst = dst as *mut HeapBin;
                let new_bin_ptr = dst.add(1) as *mut u8;
                // Copy referenced part of binary to heap
                ptr::copy_nonoverlapping(bin_ptr, new_bin_ptr, bin_size);
                // Write heapbin header
                ptr::write(dst, HeapBin::from_raw_parts(bin_header, bin_flags));
                // Write a move marker to the original location
                let marker = Term::make_boxed(heap_top);
                ptr::write(orig, marker);
                // Update `ptr` as well
                ptr::write(ptr, marker);
                // Update top pointer
                self.top = new_bin_ptr.add(bin_size) as *mut Term;
                // We're done
                return 1 + to_word_size(mem::size_of::<HeapBin>() + bin_size);
            }
        }

        let num_elements = header.arityval();
        let moved = num_elements + 1;

        let marker = Term::make_boxed(heap_top);
        // Write the term header to the location pointed to by the boxed term
        ptr::write(heap_top, header);
        // Move `ptr` to the first arityval
        // Move heap_top to the first data location
        // Then copy arityval data to new location
        ptr::copy_nonoverlapping(ptr.add(1), heap_top.add(1), num_elements);
        // In debug, verify that the src and dst are bitwise-equal
        debug_assert!(compare_bytes(heap_top, ptr, num_elements));
        // Write a move marker to the original location
        ptr::write(orig, marker);
        // And to `ptr` as well
        ptr::write(ptr, marker);
        // Update the top pointer
        self.top = heap_top.add(1 + num_elements);
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
        // New list value
        let list = Term::make_list(location);
        // Redirect original reference
        ptr::write(orig, list);
        // Store forwarding indicator/address
        let marker = Cons::new(Term::NONE, list);
        ptr::write(ptr as *mut Cons, marker);
        // Update the top pointer
        self.top = location.add(1) as *mut Term;
    }

    /// This function is used during garbage collection to sweep this heap for references
    /// that reside in younger generation heaps and have not yet been moved to this heap.
    ///
    /// In more general terms, this function will walk the current heap until
    /// the top of the heap is reached, searching for any values outside this heap
    /// heap that require moving.
    pub fn sweep(&mut self, young: &mut YoungHeap) {
        self.walk(|heap: &mut Self, term: Term, pos: *mut Term| {
            unsafe {
                if term.is_boxed() {
                    let ptr = term.boxed_val();
                    let boxed = *ptr;
                    if is_move_marker(boxed) {
                        assert!(boxed.is_boxed());
                        // Overwrite move marker with "real" boxed term
                        ptr::write(pos, boxed);
                    } else if !term.is_literal() && !heap.contains(ptr) {
                        if boxed.is_procbin() {
                            // First we need to remove the procbin from its old virtual heap
                            let old_bin = &*(ptr as *mut ProcBin);
                            young.virtual_heap_unlink(old_bin);
                            // Move to top of this heap
                            heap.move_into(pos, ptr, boxed);
                            // Then add the procbin to the new virtual heap
                            let marker = *ptr;
                            assert!(marker.is_boxed());
                            let bin_ptr = marker.boxed_val() as *mut ProcBin;
                            let bin = &*bin_ptr;
                            heap.virtual_alloc(bin);
                        } else {
                            // Move to top of this heap
                            heap.move_into(pos, ptr, boxed);
                        }
                    }
                    Some(pos.add(1))
                } else if term.is_non_empty_list() {
                    let ptr = term.list_val();
                    let cons = *ptr;
                    if cons.is_move_marker() {
                        // Overwrite move marker with "real" list term
                        ptr::write(pos, cons.tail);
                    } else if !term.is_literal() && !heap.contains(ptr) {
                        // Move to top of this heap
                        heap.move_cons_into(pos, ptr, cons);
                    }
                    Some(pos.add(1))
                } else if term.is_header() {
                    if term.is_tuple_header() {
                        // We need to check all elements, so we just skip over the tuple header
                        Some(pos.add(1))
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
                        } else if !orig_term.is_literal() && !heap.contains(ptr) {
                            heap.move_into(orig, ptr, bin);
                            ptr::write(base, binary_bytes(bin));
                        }
                        Some(pos.add(1 + term.arityval()))
                    } else {
                        Some(pos.add(1 + term.arityval()))
                    }
                } else {
                    Some(pos.add(1))
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
impl Drop for OldHeap {
    fn drop(&mut self) {
        unsafe {
            if self.active() {
                // Free virtual binary heap, we can't free the memory of this heap until we've done
                // this
                self.vheap.clear();
                // Free memory region managed by this heap instance
                process::alloc::free(self.start, self.size());
            }
        }
    }
}
impl fmt::Debug for OldHeap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!(
            "OldHeap (heap: {:?}-{:?}, used: {}, unused: {}) [\n",
            self.start,
            self.top,
            self.heap_used(),
            self.heap_available(),
        ))?;
        let mut pos = self.start;
        while pos < self.top {
            unsafe {
                let term = &*pos;
                if term.is_immediate() || term.is_boxed() || term.is_non_empty_list() {
                    f.write_fmt(format_args!("  {:?}: {:?}\n", pos, term))?;
                    pos = pos.add(1);
                } else {
                    assert!(term.is_header());
                    let arityval = term.arityval();
                    f.write_fmt(format_args!("  {:?}: {:?}\n", pos, term))?;
                    pos = pos.add(1 + arityval);
                }
            }
        }
        f.write_fmt(format_args!("  {:?}: END OF HEAP", pos))?;
        f.write_str("]\n")
    }
}

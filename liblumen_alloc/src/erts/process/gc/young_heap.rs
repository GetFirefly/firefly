use core::alloc::AllocErr;
use core::mem;
use core::ptr::{self, NonNull};

use liblumen_core::util::pointer::{distance_absolute, in_area};

use super::*;
use crate::erts::*;

/// This struct represents the current heap and stack of a process,
/// which corresponds to the young generation in the overall garbage
/// collection scheme.
///
/// This heap also includes management of the virtual binary heap, which
/// is used to track off-heap binaries (i.e. `ProcBin`) as part of the
/// overall size of the process heap in determining when garbage collection
/// is needed.
///
/// The `high_water_mark` value is used to track where the last garbage collection
/// ended, and is used to determine whether a given term is copied to a new
/// young heap, or to the old heap. This corresponds to the generational hypothesis,
/// i.e. if a term is below the high-water mark, then it has survived at least one
/// collection, and therefore is likely to be long-lived, and thus should be placed
/// in the old heap; while terms above the high-water mark have not survived a
/// collection yet, and are either garbage to be collected, or values which need to
/// be copied to the new young heap.
#[derive(Debug)]
pub struct YoungHeap {
    pub(in crate::erts::process) start: *mut Term,
    pub(in crate::erts::process) top: *mut Term,
    pub(in crate::erts::process) end: *mut Term,
    pub(in crate::erts::process) stack_start: *mut Term,
    pub(in crate::erts::process) stack_end: *mut Term,
    pub(in crate::erts::process) high_water_mark: *mut Term,
}
impl YoungHeap {
    #[inline]
    pub fn new(start: *mut Term, size: usize) -> Self {
        let end = unsafe { start.offset(size as isize) };
        let top = start;
        let stack_end = end;
        let stack_start = stack_end;
        Self {
            start,
            end,
            top,
            stack_start,
            stack_end,
            high_water_mark: start,
        }
    }

    /// Gets the total size of allocated to this heap (in words)
    #[inline]
    pub fn size(&self) -> usize {
        distance_absolute(self.end, self.start)
    }

    /// Gets the current amount of heap space used (in words)
    #[inline]
    pub fn heap_used(&self) -> usize {
        distance_absolute(self.top, self.start)
    }

    /// Gets the current amount of stack space used (in words)
    #[inline]
    pub fn stack_used(&self) -> usize {
        distance_absolute(self.stack_end, self.stack_start)
    }

    /// Gets the current amount of unused space (in words)
    ///
    /// Unused space is available to both heap and stack allocations
    #[inline]
    pub fn unused(&self) -> usize {
        distance_absolute(self.stack_start, self.top)
    }

    /// Gets the current amount of space (in words) available for heap allocations
    #[inline]
    pub fn heap_available(&self) -> usize {
        self.unused()
    }

    /// Gets the current amount of space (in words) available for stack allocations
    #[inline]
    pub fn stack_available(&self) -> usize {
        self.unused()
    }

    /// Returns the size of the mature region, i.e. terms below the high water mark
    #[inline]
    pub(crate) fn mature_size(&self) -> usize {
        distance_absolute(self.high_water_mark, self.start)
    }

    /// Returns true if the given pointer points into this heap
    #[allow(unused)]
    #[inline]
    pub fn contains<T>(&self, term: *const T) -> bool {
        in_area(term, self.start, self.top)
    }

    /// Sets the high water mark to the current top of the heap
    #[inline]
    pub fn set_high_water_mark(&mut self) {
        self.high_water_mark = self.top;
    }

    /// Perform a heap allocation of `size` words (i.e. size of `Term`)
    ///
    /// Returns `Err(AllocErr)` if there is not enough space available
    #[inline]
    pub fn alloc(&mut self, size: usize) -> Result<NonNull<Term>, AllocErr> {
        if self.heap_available() >= size {
            let ptr = self.top;
            self.top = unsafe { self.top.offset(size as isize) };
            Ok(unsafe { NonNull::new_unchecked(ptr) })
        } else {
            Err(AllocErr)
        }
    }

    /// Perform a stack allocation of `size` words (i.e. size of `Term`)
    ///
    /// Returns `Err(AllocErr)` if there is not enough space available
    #[inline]
    pub fn stack_alloc(&mut self, size: usize) -> Result<NonNull<Term>, AllocErr> {
        if self.stack_available() >= size {
            let ptr = self.stack_start;
            self.stack_start = unsafe { self.stack_start.offset(-(size as isize)) };
            Ok(unsafe { NonNull::new_unchecked(ptr) })
        } else {
            Err(AllocErr)
        }
    }

    /// This function "pops" the last `size` words from the stack, making that
    /// space available for new stack allocations.
    ///
    /// # Safety
    ///
    /// This function must be used with care, as it is intended to mirror `alloca`
    /// in the sense that allocations are "popped" in the reverse order that they
    /// occur. If this function is called with an arbitrary `size` that does not match
    /// that order, then depending on what values are stored on the stack, you may
    /// incorrectly free portions of a term which will now result in undefined behavior
    /// when that term is accessed, e.g. freeing half of a tuple.
    #[inline]
    pub fn stack_pop(&mut self, size: usize) {
        self.stack_start = unsafe { self.stack_start.offset(size as isize) };
        assert!(self.stack_start as usize <= self.stack_end as usize);
    }

    /// Copies the stack from another `YoungHeap` into this heap
    ///
    /// NOTE: This function will panic if the stack size of `other`
    /// exceeds the unused space available in this heap
    #[inline]
    pub fn copy_stack_from(&mut self, other: &Self) {
        let stack_size = other.stack_used();
        assert!(stack_size <= self.unused());
        // Determine new stack start pointer, then copy the stack
        let stack_start = unsafe { self.stack_end.offset(-(stack_size as isize)) };
        unsafe { ptr::copy_nonoverlapping(other.stack_start, stack_start, stack_size) };
        // Set stack_start
        self.stack_start = stack_start;
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
    /// heap.walk(|term, pos| {
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
        F: FnMut(&mut YoungHeap, Term, *mut Term) -> Option<*mut Term>,
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
                ptr::write(
                    dst,
                    HeapBin::from_raw_parts(bin_header, bin_flags),
                );
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
    /// that reside in the fromspace heap and have not yet been moved to this heap. At
    /// this point, the root set at a minimum will have been moved to this heap, however
    /// the moved values themselves can contain references to data which has not been moved
    /// yet, which is what this function is responsible for doing.
    ///
    /// In more general terms however, this function will walk the current heap until
    /// the top of the heap is reached, searching for any values in the fromspace
    /// heap that require moving. A reference to the old generation heap is also provided
    /// so that we can check that a candidate pointer is not in the old generation already,
    /// as those values are out of scope for this sweep
    pub fn sweep(&mut self, old: &OldHeap) {
        self.walk(|heap: &mut Self, term: Term, pos: *mut Term| {
            unsafe {
                if term.is_boxed() {
                    let ptr = term.boxed_val();
                    let boxed = *ptr;
                    if is_move_marker(boxed) {
                        assert!(boxed.is_boxed());
                        // Overwrite move marker with "real" boxed term
                        ptr::write(pos, boxed);
                    } else if in_young_gen(ptr, boxed, old) {
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
                    } else if !old.contains(ptr) {
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
                        } else if in_young_gen(ptr, bin, old) {
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

    /// Essentially the same as `sweep`, except it always moves values into
    /// this heap, since the goal is to consolidate all generations into a
    /// fresh young heap, which is this heap when called
    pub fn full_sweep(&mut self) {
        self.walk(|heap: &mut Self, term: Term, pos: *mut Term| {
            unsafe {
                if term.is_boxed() {
                    let ptr = term.boxed_val();
                    let boxed = *ptr;
                    if is_move_marker(boxed) {
                        assert!(boxed.is_boxed());
                        // Overwrite move marker with "real" boxed term
                        ptr::write(pos, boxed);
                    } else if !boxed.is_literal() {
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
                    } else {
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
                        } else {
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

    #[cfg(debug_assertions)]
    #[inline]
    pub(crate) fn sanity_check(&self) {
        let hb = self.start as usize;
        let st = self.stack_end as usize;
        let size = self.size() * mem::size_of::<Term>();
        assert!(
            hb < st,
            "bottom of the heap must be a lower address than the end of the stack"
        );
        assert!(size == st - hb, "mismatch between heap size and the actual distance between the start of the heap and end of the stack");
        let ht = self.top as usize;
        assert!(
            hb <= ht,
            "bottom of the heap must be a lower address than or equal to the top of the heap"
        );
        let sb = self.stack_start as usize;
        assert!(
            ht <= sb,
            "top of the heap must be a lower address than or equal to the start of the stack"
        );
        assert!(
            sb <= st,
            "start of the stack must be a lower address than or equal to the end of the stack"
        );
        let hwm = self.high_water_mark as usize;
        assert!(
            hb <= hwm,
            "bottom of the heap must be a lower address than or equal to the high water mark"
        );
        assert!(
            hwm <= st,
            "high water mark must be a lower address than or equal to the end of the stack"
        );
        self.overrun_check();
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    pub(crate) fn sanity_check(&self) {
        self.overrun_check();
    }

    #[inline]
    fn overrun_check(&self) {
        if (self.stack_start as usize) < (self.top as usize) {
            panic!(
                "Detected overrun of stack/heap: stack_start = {:?}, stack_end = {:?}, heap_start = {:?}, heap_top = {:?}",
                self.stack_start,
                self.stack_end,
                self.start,
                self.top,
            );
        }
    }
}

/// This function determines if the given term is located in the young generation heap
///
/// The term is provided along with its pointer, as well as a reference to
/// the old generation heap, as any pointer into the old generation cannot be
/// a young value by definition
#[inline]
pub(crate) fn in_young_gen<T>(ptr: *mut T, term: Term, old: &OldHeap) -> bool {
    !term.is_literal() && !old.contains(ptr)
}

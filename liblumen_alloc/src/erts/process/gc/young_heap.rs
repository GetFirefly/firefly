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
    pub(in crate::erts::process) stack_size: usize,
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
            stack_size: 0,
            high_water_mark: start,
        }
    }

    /// Gets the total size of allocated to this heap (in words)
    #[inline]
    pub fn size(&self) -> usize {
        distance_absolute(self.end, self.start)
    }

    /// Gets the number of terms currently allocated on the stack
    #[inline]
    pub fn stack_size(&self) -> usize {
        self.stack_size
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

    /// Perform a stack allocation of `size` words to hold a single term.
    ///
    /// Returns `Err(AllocErr)` if there is not enough space available
    /// 
    /// NOTE: Do not use this to allocate space for multiple terms (lists
    /// and boxes count as a single term), as the size of the stack in terms
    /// is tied to allocations. Each time `stack_alloc` is called, the stack
    /// size is incremented by 1, and this enables efficient implementations
    /// of the other stack manipulation functions as the stack size in terms
    /// does not have to be recalculated constantly.
    #[inline]
    pub fn stack_alloc(&mut self, size: usize) -> Result<NonNull<Term>, AllocErr> {
        if self.stack_available() >= size {
            let ptr = self.stack_start;
            self.stack_start = unsafe { self.stack_start.offset(-(size as isize)) };
            self.stack_size += 1;
            Ok(unsafe { NonNull::new_unchecked(ptr) })
        } else {
            Err(AllocErr)
        }
    }

    /// This function returns the term located in the given stack slot, if it exists.
    /// 
    /// The stack slots are 1-indexed, where `1` is the top of the stack, or most recent
    /// allocation, and `S` is the bottom of the stack, or oldest allocation. 
    /// 
    /// If `S > stack_size`, then `None` is returned. Otherwise, `Some(Term)` is returned.
    #[inline]
    pub fn stack_slot(&mut self, n: usize) -> Option<Term> {
        assert!(n > 0);
        if n <= self.stack_size {
            let stack_slot_ptr = self.stack_slot_address(n - 1);
            Some(unsafe { *stack_slot_ptr })
        } else {
            None
        }
    }

    /// This function "pops" the last `n` terms from the stack, making that
    /// space available for new stack allocations.
    ///
    /// # Safety
    ///
    /// This function will panic if given a value `n` which exceeds the current
    /// number of terms allocated on the stack
    #[inline]
    pub fn stack_popn(&mut self, n: usize) {
        assert!(n > 0 && n <= self.stack_size);
        if self.stack_size - n == 0 {
            // Freeing the whole stack
            self.stack_start = self.stack_end;
            self.stack_size = 0;
        } else {
            // Freeing a subset
            let start_slot_ptr = self.stack_slot_address(n - 1);
            self.stack_start = start_slot_ptr;
            self.stack_size -= n;
        }
    }

    #[inline]
    fn stack_slot_address(&self, mut slot: usize) -> *mut Term {
        assert!(slot < self.stack_size);
        // Essentially a no-op
        if slot == 0 {
            return self.stack_start;
        }
        // Walk the stack from start to finish, where the start is the lowest
        // address. This means we walk terms from the "top" of the stack to the
        // bottom, or put another way, from the most recently pushed to the oldest.
        //
        // When terms are allocated on the stack, the stack grows downwards, but we
        // lay the term out in memory normally, i.e. allocating upwards. So scanning
        // the stack requires following terms upwards, in order for us to skip over
        // non-term data on the stack.
        let mut pos = self.stack_start;
        while slot > 0 {
            let term = unsafe { *pos };
            if term.is_immediate() {
                // Immediates are the typical case, and only occupy one word
                slot -= 1;
                pos = unsafe { pos.offset(1) };
            } else if term.is_boxed() {
                // Boxed terms will consist of the box itself, and if stored on the stack, the
                // boxed value will follow immediately afterward. The header of that value will
                // contain the size in words of the data, which we can use to skip over to the next 
                // term. In the case where the value resides on the heap, we can treat the box like
                // an immediate
                let ptr = term.boxed_val();
                pos = unsafe { pos.offset(1) };
                if ptr == pos {
                    // The boxed value is on the stack immediately following the box
                    let val = unsafe { *ptr };
                    assert!(val.is_header());
                    let skip = val.arityval() as isize;
                    slot -= 1;
                    pos = unsafe { pos.offset(skip) };
                } else {
                    assert!(!in_area(ptr, pos, self.stack_end), "boxed term stored on stack but not contiguously!");
                }
            } else if term.is_list() {
                // Lists are essentially boxes which point to cons cells, but are a bit more complicated
                // than boxed terms. Proper lists will have a cons cell where the head is nil, and improper
                // lists will have a tail that contains a non-list term. For lists entirely on the stack, they
                // may only consist of immediates or boxes which point to values on the heap, as it introduces
                // unnecessary complexity to lay out cons cells in memory where the head term is larger than one
                // word. This constraint also makes allocating lists on the stack easier to reason about.
                let ptr = term.list_val() as *mut Cons;
                let mut next_ptr = unsafe { pos.offset(1) as *mut Cons };
                // If true, the first cell is correctly laid out contiguously with the list term
                if ptr == next_ptr {
                    // This is used to hold the current cons cell
                    let mut cons = unsafe { *ptr };
                    // This is a pointer to the next location in memory following this cell
                    next_ptr = unsafe { next_ptr.offset(1) };
                    loop {
                        if cons.head.is_nil() {
                            // We've hit the end of a proper list, update `pos` and break out
                            pos = next_ptr as *mut Term;
                            break;
                        } 
                        assert!(cons.head.is_immediate() || cons.head.is_boxed(), "invalid stack-allocated list, elements must be an immediate or box");
                        if cons.tail.is_list() {
                            // The list continues, we need to check where it continues on the stack
                            let next_cons = cons.tail.list_val();
                            if next_cons == next_ptr {
                                // Yep, it is on the stack, contiguous with this cell
                                cons = unsafe { *next_ptr };
                                next_ptr = unsafe { next_ptr.offset(1) };
                                continue;
                            } else {
                                // It must be on the heap, otherwise we've violated an invariant
                                assert!(!in_area(next_cons, next_ptr as *const Term, self.stack_end), "list stored on stack but not contiguously!");
                            }
                        } else if cons.tail.is_immediate() {
                            // We've hit the end of an improper list, update `pos` and break out
                            pos = next_ptr as *mut Term;
                            break;
                        } else if cons.tail.is_boxed() {
                            // We've hit the end of an improper list, update `pos` and break out
                            let box_ptr = cons.tail.boxed_val();
                            assert!(!in_area(box_ptr, next_ptr as *const Term, self.stack_end), "invalid stack-allocated list, elements must be an immediate or box");
                            pos = next_ptr as *mut Term;
                            break;
                        }
                    }
                    slot -= 1;
                } else {
                    assert!(!in_area(ptr, pos, self.stack_end), "list term stored on stack but not contiguously!");
                }
            } else {
                unreachable!();
            }
        }
        pos
    }

    /// Copies the stack from another `YoungHeap` into this heap
    ///
    /// NOTE: This function will panic if the stack size of `other`
    /// exceeds the unused space available in this heap
    #[inline]
    pub fn copy_stack_from(&mut self, other: &Self) {
        let stack_used = other.stack_used();
        assert!(stack_used <= self.unused());
        // Determine new stack start pointer, then copy the stack
        let stack_start = unsafe { self.stack_end.offset(-(stack_used as isize)) };
        unsafe { ptr::copy_nonoverlapping(other.stack_start, stack_start, stack_used) };
        // Set stack_start
        self.stack_start = stack_start;
        self.stack_size = other.stack_size;
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

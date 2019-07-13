use core::alloc::AllocErr;
use core::fmt;
use core::mem;
use core::ptr::{self, NonNull};

use liblumen_core::util::pointer::{distance_absolute, in_area, in_area_inclusive};

use super::*;
use crate::erts::process::alloc::{HeapAlloc, StackAlloc, StackPrimitives, VirtualAlloc};
use crate::erts::term::{
    binary_bytes, is_move_marker, Cons, HeapBin, MatchContext, ProcBin, SubBinary,
};
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
pub struct YoungHeap {
    start: *mut Term,
    top: *mut Term,
    end: *mut Term,
    stack_start: *mut Term,
    stack_end: *mut Term,
    stack_size: usize,
    high_water_mark: *mut Term,
    vheap: VirtualBinaryHeap,
}
impl YoungHeap {
    /// This function creates a new `YoungHeap` which owns the given memory region
    /// represented by the pointer `start` and the size in words of the region.
    #[inline]
    pub fn new(start: *mut Term, size: usize) -> Self {
        let end = unsafe { start.offset(size as isize) };
        let top = start;
        let stack_end = end;
        let stack_start = stack_end;
        let vheap = VirtualBinaryHeap::new(size);
        Self {
            start,
            end,
            top,
            stack_start,
            stack_end,
            stack_size: 0,
            high_water_mark: start,
            vheap,
        }
    }
}
impl HeapAlloc for YoungHeap {
    #[inline]
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        if self.heap_available() >= need {
            let ptr = self.top;
            self.top = self.top.offset(need as isize);
            Ok(NonNull::new_unchecked(ptr))
        } else {
            Err(AllocErr)
        }
    }

    #[inline]
    fn is_owner<T>(&mut self, ptr: *const T) -> bool {
        self.contains(ptr) || self.vheap.contains(ptr)
    }
}
impl VirtualAlloc for YoungHeap {
    #[inline]
    fn virtual_alloc(&mut self, bin: &ProcBin) -> Term {
        self.vheap.push(bin)
    }
}
impl StackAlloc for YoungHeap {
    #[inline]
    unsafe fn alloca(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        if self.stack_available() >= need {
            Ok(self.alloca_unchecked(need))
        } else {
            Err(AllocErr)
        }
    }

    #[inline]
    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        self.stack_start = self.stack_start.offset(-(need as isize));
        self.stack_size += 1;
        NonNull::new_unchecked(self.stack_start)
    }
}
impl StackPrimitives for YoungHeap {
    #[inline(always)]
    fn stack_size(&self) -> usize {
        self.stack_size
    }

    #[inline(always)]
    unsafe fn set_stack_size(&mut self, size: usize) {
        self.stack_size = size;
    }

    #[inline(always)]
    fn stack_pointer(&mut self) -> *mut Term {
        self.stack_start
    }

    #[inline]
    unsafe fn set_stack_pointer(&mut self, sp: *mut Term) {
        assert!(
            in_area_inclusive(sp, self.top, self.stack_end),
            "attempted to set stack pointer out of bounds"
        );
        self.stack_start = sp;
    }

    /// Gets the current amount of stack space used (in words)
    #[inline]
    fn stack_used(&self) -> usize {
        distance_absolute(self.stack_end, self.stack_start)
    }

    /// Gets the current amount of space (in words) available for stack allocations
    #[inline]
    fn stack_available(&self) -> usize {
        self.unused()
    }

    #[inline]
    fn stack_slot(&mut self, n: usize) -> Option<Term> {
        assert!(n > 0);
        if n <= self.stack_size {
            let stack_slot_ptr = self.stack_slot_address(n - 1);
            Some(unsafe { *stack_slot_ptr })
        } else {
            None
        }
    }

    #[inline]
    fn stack_popn(&mut self, n: usize) {
        assert!(n > 0 && n <= self.stack_size);
        if self.stack_size - n == 0 {
            // Freeing the whole stack
            self.stack_start = self.stack_end;
            self.stack_size = 0;
        } else {
            // Freeing a subset
            let start_slot_ptr = self.stack_slot_address(n);
            debug_assert!(start_slot_ptr as usize > self.stack_start as usize);
            self.stack_start = start_slot_ptr;
            self.stack_size -= n;
        }
    }
}
impl YoungHeap {
    /// This function is used to reallocate the memory region this heap was originally allocated
    /// with to a smaller size, given by `new_size`. This function will panic if the given size
    /// is not large enough to hold the stack, if a size greater than the previous size is
    /// given, or if reallocation in place fails.
    ///
    /// On success, the heap metadata is updated to reflect the new size
    pub(super) fn shrink(&mut self, new_size: usize) {
        let total_size = self.size();
        assert!(
            new_size < total_size,
            "tried to shrink a heap with a new size that is larger than the old size"
        );
        let stack_size = self.stack_used();
        assert!(
            new_size > stack_size,
            "cannot shrink heap to be smaller than its stack usage"
        );

        // Calculate the new start (or "top") of the stack, this will be our destination pointer
        let old_start = self.start;
        let old_stack_start = self.stack_start;
        let new_stack_start = unsafe { old_start.offset((new_size - stack_size) as isize) };

        // Copy the stack into its new position
        unsafe { ptr::copy(old_stack_start, new_stack_start, stack_size) };

        // Reallocate the heap to shrink it, if the heap is moved, there is a bug
        // in the allocator which must have been introduced in a recent change
        let new_heap = unsafe {
            process::alloc::realloc(old_start, total_size, new_size)
                .expect("unable to shrink heap memory for process: realloc failed!")
        };
        assert_eq!(
            new_heap, old_start,
            "expected reallocation of heap during shrink to occur in-place!"
        );

        self.end = unsafe { new_heap.offset(new_size as isize) };
        self.stack_end = self.end;
        self.stack_start = unsafe { self.stack_end.offset(-(stack_size as isize)) };
    }

    /// Gets the total size of allocated to this heap (in words)
    #[inline]
    pub fn size(&self) -> usize {
        distance_absolute(self.end, self.start)
    }

    /// Returns the pointer to the bottom of the heap
    #[inline(always)]
    pub fn heap_start(&self) -> *mut Term {
        self.start
    }

    /// Gets the current amount of heap space used (in words)
    #[inline]
    pub fn heap_used(&self) -> usize {
        distance_absolute(self.top, self.start)
    }

    /// Gets the current amount of unused space (in words)
    ///
    /// Unused space is available to both heap and stack allocations
    #[inline]
    pub fn unused(&self) -> usize {
        distance_absolute(self.stack_start, self.top)
    }

    /// Returns the used size of the virtual heap
    #[inline]
    pub fn virtual_heap_used(&self) -> usize {
        self.vheap.heap_used()
    }

    /// Returns the unused size of the virtual heap
    #[inline]
    pub fn virtual_heap_unused(&self) -> usize {
        self.vheap.unused()
    }

    /// Gets the current amount of space (in words) available for heap allocations
    #[inline]
    pub fn heap_available(&self) -> usize {
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

    #[inline]
    fn stack_slot_address(&self, slot: usize) -> *mut Term {
        assert!(slot < self.stack_size);
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
        let mut last = self.stack_start;
        let mut pos = last;
        for _ in 0..=slot {
            let term = unsafe { *pos };
            if term.is_immediate() {
                // Immediates are the typical case, and only occupy one word
                last = pos;
                pos = unsafe { pos.offset(1) };
            } else if term.is_boxed() {
                // Boxed terms will consist of the box itself, and if stored on the stack, the
                // boxed value will follow immediately afterward. The header of that value will
                // contain the size in words of the data, which we can use to skip over to the next
                // term. In the case where the value resides on the heap, we can treat the box like
                // an immediate
                let ptr = term.boxed_val();
                last = pos;
                pos = unsafe { pos.offset(1) };
                if ptr == pos {
                    // The boxed value is on the stack immediately following the box
                    let val = unsafe { *ptr };
                    assert!(val.is_header());
                    let skip = val.arityval() as isize;
                    pos = unsafe { pos.offset(skip) };
                } else {
                    assert!(
                        !in_area(ptr, pos, self.stack_end),
                        "boxed term stored on stack but not contiguously!"
                    );
                }
            } else if term.is_list() {
                // The list begins here
                last = pos;
                pos = unsafe { pos.offset(1) };
                // Lists are essentially boxes which point to cons cells, but are a bit more
                // complicated than boxed terms. Proper lists will have a cons cell
                // where the head is nil, and improper lists will have a tail that
                // contains a non-list term. For lists entirely on the stack, they
                // may only consist of immediates or boxes which point to values on the heap, as it
                // introduces unnecessary complexity to lay out cons cells in memory
                // where the head term is larger than one word. This constraint also
                // makes allocating lists on the stack easier to reason about.
                let ptr = term.list_val() as *mut Cons;
                let mut next_ptr = pos as *mut Cons;
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
                        assert!(
                            cons.head.is_immediate() || cons.head.is_boxed(),
                            "invalid stack-allocated list, elements must be an immediate or box"
                        );
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
                                assert!(
                                    !in_area(next_cons, next_ptr as *const Term, self.stack_end),
                                    "list stored on stack but not contiguously!"
                                );
                            }
                        } else if cons.tail.is_immediate() {
                            // We've hit the end of the list, update `pos` and break out
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
                } else {
                    assert!(
                        !in_area(ptr, pos, self.stack_end),
                        "list term stored on stack but not contiguously!"
                    );
                }
            } else {
                unreachable!();
            }
        }
        last
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
        debug_assert_ne!(orig, self.top);

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
                let val = Term::make_boxed(dst);
                ptr::write(heap_top, val);
                let dst = dst as *mut HeapBin;
                let new_bin_ptr = dst.offset(1) as *mut u8;
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
                self.top = new_bin_ptr.offset(bin_size as isize) as *mut Term;
                // We're done
                return 1 + to_word_size(mem::size_of::<HeapBin>() + bin_size);
            }
        }

        let num_elements = header.arityval();
        let moved = num_elements + 1;

        let marker = Term::make_boxed(heap_top);
        // Write the term header to the location pointed to by the boxed term
        ptr::write(heap_top, header);
        // Write the move marker to the original location
        ptr::write(orig, marker);
        // And to `ptr` as well
        ptr::write(ptr, marker);
        // Move `ptr` to the first arityval
        // Move heap_top to the first data location
        // Then copy arityval data to new location
        heap_top = heap_top.offset(1);
        ptr::copy_nonoverlapping(ptr.offset(1), heap_top, num_elements);
        // Update the top pointer
        self.top = heap_top.offset(num_elements as isize);
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
    pub fn sweep(
        &mut self,
        prev: &mut YoungHeap,
        old: &mut OldHeap,
        mature: *mut Term,
        mature_end: *mut Term,
    ) {
        self.walk(|heap: &mut Self, term: Term, pos: *mut Term| {
            unsafe {
                if term.is_boxed() {
                    let ptr = term.boxed_val();
                    let boxed = *ptr;
                    if is_move_marker(boxed) {
                        assert!(boxed.is_boxed());
                        // Overwrite move marker with "real" boxed term
                        ptr::write(pos, boxed);
                    } else if in_area(ptr, mature, mature_end) {
                        // Move into old generation
                        if boxed.is_procbin() {
                            // First we need to remove the procbin from its old virtual heap
                            let old_bin = &*(ptr as *mut ProcBin);
                            if prev.virtual_heap_contains(old_bin) {
                                prev.virtual_heap_unlink(old_bin);
                            } else {
                                heap.virtual_heap_unlink(old_bin);
                            }
                            // Move to top of the old gen heap
                            old.move_into(pos, ptr, boxed);
                            // Then add the procbin to the old gen virtual heap
                            let marker = *ptr;
                            assert!(marker.is_boxed());
                            let bin_ptr = marker.boxed_val() as *mut ProcBin;
                            let bin = &*bin_ptr;
                            old.virtual_alloc(bin);
                        } else {
                            old.move_into(pos, ptr, boxed);
                        }
                    } else if !term.is_literal() && !old.contains(ptr) && !heap.contains(ptr) {
                        // Move into young generation (this heap)
                        if boxed.is_procbin() {
                            // First we need to remove the procbin from its old virtual heap
                            let old_bin = &*(ptr as *mut ProcBin);
                            prev.virtual_heap_unlink(old_bin);
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
                    Some(pos.offset(1))
                } else if term.is_non_empty_list() {
                    let ptr = term.list_val();
                    let cons = *ptr;
                    if cons.is_move_marker() {
                        // Overwrite move marker with "real" list term
                        ptr::write(pos, cons.tail);
                    } else if in_area(ptr, mature, mature_end) {
                        // Move to old generation
                        old.move_cons_into(pos, ptr, cons);
                    } else if !term.is_literal() && !old.contains(ptr) && !heap.contains(ptr) {
                        // Move to top of this heap
                        heap.move_cons_into(pos, ptr, cons);
                    }
                    Some(pos.offset(1))
                } else if term.is_header() {
                    if term.is_tuple_header() {
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
                        } else if in_area(ptr, mature, mature_end) {
                            // Move to old generation
                            old.move_into(orig, ptr, bin);
                            ptr::write(base, binary_bytes(bin));
                        } else if !orig_term.is_literal() && !old.contains(ptr) {
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
    pub fn full_sweep(&mut self, prev: &mut YoungHeap, old: &mut OldHeap) {
        self.walk(|heap: &mut Self, term: Term, pos: *mut Term| {
            unsafe {
                if term.is_boxed() {
                    let ptr = term.boxed_val();
                    let boxed = *ptr;
                    if is_move_marker(boxed) {
                        assert!(boxed.is_boxed());
                        // Overwrite move marker with "real" boxed term
                        ptr::write(pos, boxed);
                    } else if !term.is_literal() {
                        if boxed.is_procbin() {
                            // First we need to remove the procbin from its old virtual heap
                            let old_bin = &*(ptr as *mut ProcBin);
                            if old.virtual_heap_contains(old_bin) {
                                old.virtual_heap_unlink(old_bin);
                            } else {
                                prev.virtual_heap_unlink(old_bin);
                            }
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
                    // Move past box pointer to next term
                    Some(pos.offset(1))
                } else if term.is_non_empty_list() {
                    let ptr = term.list_val();
                    let cons = *ptr;
                    if cons.is_move_marker() {
                        assert!(cons.tail.is_list());
                        // Rewrite marker with list pointer
                        ptr::write(pos, cons.tail);
                    } else if !term.is_literal() {
                        // Move cons cell to top of this heap
                        heap.move_cons_into(pos, ptr, cons);
                    }
                    // Move past list pointer to next term
                    Some(pos.offset(1))
                } else if term.is_header() {
                    if term.is_tuple_header() {
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
                        } else if !orig_term.is_literal() {
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
    pub(super) fn sanity_check(&self) {
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
    pub(super) fn sanity_check(&self) {
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
impl Drop for YoungHeap {
    fn drop(&mut self) {
        unsafe {
            // Free virtual binary heap
            self.vheap.clear();
            // Free memory region managed by this heap instance
            process::alloc::free(self.start, self.size());
        }
    }
}
impl fmt::Debug for YoungHeap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!(
            "YoungHeap (heap: {:?}-{:?}, stack: {:?}-{:?}, used: {}, unused: {}) [\n",
            self.start,
            self.top,
            self.stack_start,
            self.stack_end,
            self.heap_used() + self.stack_used(),
            self.unused(),
        ))?;
        let mut pos = self.start;
        while pos < self.top {
            unsafe {
                let term = *pos;
                if term.is_immediate() || term.is_boxed() || term.is_list() {
                    f.write_fmt(format_args!("  {:?}: {:?}\n", pos, term))?;
                    pos = pos.offset(1);
                } else {
                    assert!(term.is_header());
                    let arityval = term.arityval();
                    f.write_fmt(format_args!("  {:?}: {:?}\n", pos, term))?;
                    pos = pos.offset((1 + arityval) as isize);
                }
            }
        }
        f.write_str("  ==== END HEAP ====\n")?;
        f.write_str("  ==== START STACK ==== \n")?;
        pos = self.stack_start;
        while pos < self.stack_end {
            unsafe {
                let term = *pos;
                if term.is_immediate() || term.is_boxed() || term.is_list() {
                    f.write_fmt(format_args!("  {:?}: {:?}\n", pos, term))?;
                    pos = pos.offset(1);
                } else {
                    assert!(term.is_header());
                    let arityval = term.arityval();
                    f.write_fmt(format_args!("  {:?}: {:?}\n", pos, term))?;
                    pos = pos.offset((1 + arityval) as isize);
                }
            }
        }
        f.write_str("]\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

    use crate::erts::process::alloc;

    #[test]
    fn young_heap_alloc() {
        let (heap, heap_size) = alloc::default_heap().unwrap();
        let mut yh = YoungHeap::new(heap, heap_size);

        assert_eq!(yh.size(), heap_size);
        assert_eq!(yh.stack_size(), 0);
        unsafe {
            yh.set_stack_size(3);
        }
        assert_eq!(yh.stack_size(), 3);
        unsafe {
            yh.set_stack_size(0);
        }
        assert_eq!(yh.stack_used(), 0);
        assert_eq!(yh.heap_used(), 0);
        assert_eq!(yh.unused(), heap_size);
        assert_eq!(yh.unused(), yh.heap_available());
        assert_eq!(yh.unused(), yh.stack_available());
        assert_eq!(yh.mature_size(), 0);

        let nil = unsafe { yh.alloc(1).unwrap().as_ptr() };
        assert_eq!(yh.heap_used(), 1);
        assert_eq!(yh.unused(), heap_size - 1);
        unsafe { ptr::write(nil, Term::NIL) };

        // Allocate the list `[101, "test"]`
        let num = Term::make_smallint(101);
        let string = "test";
        let string_term = make_heapbin_from_str(&mut yh, string).unwrap();
        let list_term = ListBuilder::new(&mut yh)
            .push(num)
            .push(string_term)
            .finish()
            .unwrap();
        let list_size =
            to_word_size((mem::size_of::<Cons>() * 2) + mem::size_of::<HeapBin>() + string.len());
        assert_eq!(yh.heap_used(), 1 + list_size);
        assert!(list_term.is_list());
        let list_ptr = list_term.list_val();
        let first_cell = unsafe { &*list_ptr };
        assert!(first_cell.head.is_smallint());
        assert!(first_cell.tail.is_list());
        let tail_ptr = first_cell.tail.list_val();
        let second_cell = unsafe { &*tail_ptr };
        assert!(second_cell.head.is_boxed());
        let bin_ptr = second_cell.head.boxed_val();
        let bin_term = unsafe { *bin_ptr };
        assert!(bin_term.is_heapbin());
        let hb = unsafe { &*(bin_ptr as *mut HeapBin) };
        assert_eq!(hb.as_str(), string);
    }

    #[test]
    fn young_heap_stack_alloc() {
        let (heap, heap_size) = alloc::default_heap().unwrap();
        let mut yh = YoungHeap::new(heap, heap_size);

        // Allocate the list `[101, :foo]` on the stack
        let num = Term::make_smallint(101);
        let foo = unsafe { Atom::try_from_str("foo").unwrap().as_term() };
        let _list_term = HeaplessListBuilder::new(&mut yh)
            .push(num)
            .push(foo)
            .finish()
            .unwrap();
        // 2 cons cells + 1 box term
        let list_size = to_word_size(mem::size_of::<Cons>() * 2) + 1;

        assert_eq!(yh.heap_used(), 0);
        assert_eq!(yh.stack_used(), list_size);
        assert_eq!(yh.stack_size(), 1);

        // Allocate a binary and put the pointer on the stack
        let string = "bar";
        let string_term = make_heapbin_from_str(&mut yh, string).unwrap();
        unsafe {
            let ptr = yh.alloca(1).unwrap().as_ptr();
            ptr::write(ptr, string_term);
        }

        // Verify stack sizes
        assert_eq!(yh.stack_used(), list_size + 1);
        assert_eq!(yh.stack_size(), 2);

        // Fetch top of stack, expect box
        let slot_term_addr = yh.stack_slot_address(0);
        assert_eq!(slot_term_addr, yh.stack_start);
        let slot_term = unsafe { *slot_term_addr };
        assert!(slot_term.is_boxed());

        // Fetch top - 1 of stack, expect list
        let slot_term_addr = yh.stack_slot_address(1);
        assert_eq!(slot_term_addr, unsafe { yh.stack_start.offset(1) });
        let slot_term = unsafe { *slot_term_addr };
        assert!(slot_term.is_list());
        let list = unsafe { *slot_term.list_val() };
        assert!(list.head.is_smallint());
        assert!(list.tail.is_list());
        let tail = unsafe { *list.tail.list_val() };
        assert!(tail.head.is_atom());
        assert!(tail.tail.is_nil());
    }
}

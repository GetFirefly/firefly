use core::alloc::Layout;
use core::fmt;
use core::mem;
use core::ptr::{self, NonNull};

use liblumen_core::alloc::utils::{align_up_to, is_aligned, is_aligned_at};
use liblumen_core::util::pointer::{distance_absolute, in_area, in_area_inclusive};

use crate::erts::exception::AllocResult;
use crate::erts::process;
use crate::erts::process::alloc::*;
use crate::erts::term::prelude::*;
use crate::mem::bit_size_of;

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
        let end = unsafe { start.add(size) };
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

    /// This function is used to reallocate the memory region this heap was originally allocated
    /// with to a smaller size, given by `new_size`. This function will panic if the given size
    /// is not large enough to hold the stack, if a size greater than the previous size is
    /// given, or if reallocation in place fails.
    ///
    /// On success, the heap metadata is updated to reflect the new size
    pub unsafe fn shrink(&mut self, new_size: usize) {
        let total_size = self.heap_size();
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
        let new_stack_start = old_start.add(new_size - stack_size);

        // Copy the stack into its new position
        ptr::copy(old_stack_start, new_stack_start, stack_size);

        // Reallocate the heap to shrink it, if the heap is moved, there is a bug
        // in the allocator which must have been introduced in a recent change
        let new_heap =
            process::alloc::realloc(old_start, total_size, new_size).unwrap_or(old_start);
        assert_eq!(
            new_heap, old_start,
            "expected reallocation of heap during shrink to occur in-place!"
        );

        self.end = new_heap.add(new_size);
        self.stack_end = self.end;
        self.stack_start = self.stack_end.offset(-(stack_size as isize));
    }

    /// Gets the current amount of unused space (in words)
    ///
    /// Unused space is available to both heap and stack allocations
    #[inline]
    pub fn unused(&self) -> usize {
        self.heap_available()
    }

    /// Returns the size of the mature region, i.e. terms below the high water mark
    #[inline]
    pub(crate) fn mature_size(&self) -> usize {
        distance_absolute(self.high_water_mark, self.start)
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
                pos = unsafe { pos.add(1) };
            } else if term.is_boxed() {
                // Boxed terms will consist of the box itself, and if stored on the stack, the
                // boxed value will follow immediately afterward. The header of that value will
                // contain the size in words of the data, which we can use to skip over to the next
                // term. In the case where the value resides on the heap, we can treat the box like
                // an immediate
                let ptr: *mut Term = term.dyn_cast();
                last = pos;
                pos = unsafe { pos.add(1) };
                if ptr == pos {
                    // The boxed value is on the stack immediately following the box
                    let val = unsafe { *ptr };
                    assert!(val.is_header());
                    let skip = val.arity();
                    pos = unsafe { pos.add(skip) };
                } else {
                    assert!(
                        !in_area(ptr, pos, self.stack_end),
                        "boxed term stored on stack but not contiguously!"
                    );
                }
            } else if term.is_non_empty_list() {
                // The list begins here
                last = pos;
                pos = unsafe { pos.add(1) };
                // Lists are essentially boxes which point to cons cells, but are a bit more
                // complicated than boxed terms. Proper lists will have a cons cell
                // where the head is nil, and improper lists will have a tail that
                // contains a non-list term. For lists entirely on the stack, they
                // may only consist of immediates or boxes which point to values on the heap, as it
                // introduces unnecessary complexity to lay out cons cells in memory
                // where the head term is larger than one word. This constraint also
                // makes allocating lists on the stack easier to reason about.
                let ptr: *mut Cons = term.dyn_cast();
                let mut next_ptr = pos as *mut Cons;
                // If true, the first cell is correctly laid out contiguously with the list term
                if ptr == next_ptr {
                    // This is used to hold the current cons cell
                    let mut cons = unsafe { *ptr };
                    // This is a pointer to the next location in memory following this cell
                    next_ptr = unsafe { next_ptr.add(1) };
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
                        if cons.tail.is_non_empty_list() {
                            // The list continues, we need to check where it continues on the stack
                            let next_cons: *mut Cons = cons.tail.dyn_cast();
                            if next_cons == next_ptr {
                                // Yep, it is on the stack, contiguous with this cell
                                cons = unsafe { *next_ptr };
                                next_ptr = unsafe { next_ptr.add(1) };
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
                            let box_ptr: *mut Term = cons.tail.dyn_cast();
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
                unreachable!("{:?} should not be in a stack slot", term);
            }
        }
        last
    }

    /// Copies the stack from another `YoungHeap` into this heap
    ///
    /// NOTE: This function will panic if the stack size of `other`
    /// exceeds the unused space available in this heap
    #[inline]
    pub(super) unsafe fn copy_stack_from(&mut self, other: &Self) {
        let stack_used = other.stack_used();
        assert!(stack_used <= self.unused());
        // Determine new stack start pointer, then copy the stack
        let stack_start = self.stack_end.offset(-(stack_used as isize));
        ptr::copy_nonoverlapping(other.stack_start, stack_start, stack_used);
        // Set stack_start
        self.stack_start = stack_start;
        self.stack_size = other.stack_size;
    }

    #[inline]
    fn overrun_check(&self) {
        if self.stack_start < self.top {
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
impl Heap for YoungHeap {
    #[inline(always)]
    fn heap_start(&self) -> *mut Term {
        self.start
    }

    #[inline(always)]
    fn heap_top(&self) -> *mut Term {
        self.top
    }

    #[inline(always)]
    fn heap_end(&self) -> *mut Term {
        self.stack_start
    }

    #[inline]
    fn heap_size(&self) -> usize {
        distance_absolute(self.end, self.start)
    }

    #[inline]
    fn heap_available(&self) -> usize {
        distance_absolute(self.stack_start, self.top)
    }

    /// Gets the high water mark
    #[inline]
    fn high_water_mark(&self) -> *mut Term {
        self.high_water_mark
    }

    #[inline]
    fn contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        in_area(ptr, self.heap_start(), self.heap_top())
    }

    #[inline]
    fn is_owner<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.contains(ptr) || self.virtual_contains(ptr)
    }

    #[cfg(debug_assertions)]
    #[inline]
    fn sanity_check(&self) {
        let hb = self.heap_start();
        let st = self.stack_end;
        let size = self.heap_size() * mem::size_of::<Term>();
        assert!(
            hb < st,
            "bottom of the heap must be a lower address than the end of the stack"
        );
        assert_eq!(size, (st as usize - hb as usize), "mismatch between heap size and the actual distance between the start of the heap and end of the stack");
        let ht = self.top;
        assert!(
            hb <= ht,
            "bottom of the heap must be a lower address than or equal to the top of the heap"
        );
        let sb = self.stack_start;
        assert!(
            ht <= sb,
            "top of the heap must be a lower address than or equal to the start of the stack"
        );
        assert!(
            sb <= st,
            "start of the stack must be a lower address than or equal to the end of the stack"
        );
        let hwm = self.high_water_mark;
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
    fn sanity_check(&self) {
        self.overrun_check();
    }
}
impl HeapAlloc for YoungHeap {
    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        use liblumen_core::sys::sysconf::MIN_ALIGN;

        let layout = layout.align_to(MIN_ALIGN).unwrap().pad_to_align();

        let needed = layout.size();
        let available = self.heap_available() * mem::size_of::<Term>();
        if needed > available {
            return Err(alloc!());
        }

        let top = self.top as *mut u8;
        let new_top = top.add(needed);
        debug_assert!(new_top <= self.end as *mut u8);
        self.top = new_top as *mut Term;

        let align = layout.align();
        let ptr = if is_aligned_at(top, align) {
            top as *mut Term
        } else {
            align_up_to(top as *mut Term, align)
        };
        // Success!
        debug_assert!(is_aligned(ptr));
        Ok(NonNull::new_unchecked(ptr))
    }
}
impl VirtualHeap<ProcBin> for YoungHeap {
    #[inline]
    fn virtual_size(&self) -> usize {
        self.vheap.virtual_size()
    }

    #[inline]
    fn virtual_heap_used(&self) -> usize {
        self.vheap.virtual_heap_used()
    }

    #[inline]
    fn virtual_heap_unused(&self) -> usize {
        self.vheap.virtual_heap_unused()
    }
}
impl VirtualAllocator<ProcBin> for YoungHeap {
    #[inline]
    fn virtual_alloc(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_alloc(value)
    }

    #[inline]
    fn virtual_free(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_free(value)
    }

    #[inline]
    fn virtual_unlink(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_unlink(value)
    }

    #[inline]
    fn virtual_pop(&mut self, value: Boxed<ProcBin>) -> ProcBin {
        self.vheap.virtual_pop(value)
    }

    #[inline]
    fn virtual_contains<P: ?Sized>(&self, ptr: *const P) -> bool {
        self.vheap.virtual_contains(ptr)
    }

    #[inline]
    unsafe fn virtual_clear(&mut self) {
        self.vheap.virtual_clear()
    }
}
impl StackAlloc for YoungHeap {
    #[inline]
    unsafe fn alloca(&mut self, need: usize) -> AllocResult<NonNull<Term>> {
        if self.stack_available() >= need {
            Ok(self.alloca_unchecked(need))
        } else {
            Err(alloc!())
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
        assert!(n > 0);
        assert!(
            n <= self.stack_size,
            "Trying to pop {} terms from stack that only has {} terms",
            n,
            self.stack_size
        );
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
impl Drop for YoungHeap {
    fn drop(&mut self) {
        unsafe {
            let heap_size = self.heap_size();
            // Free virtual binary heap
            self.virtual_clear();
            // Zero-sized heaps have no backing memory
            if heap_size > 0 {
                // Free memory region managed by this heap instance
                process::alloc::free(self.heap_start(), self.heap_size());
            }
        }
    }
}
impl fmt::Debug for YoungHeap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::erts::term::arch::repr::Repr;

        f.write_fmt(format_args!(
            "YoungHeap (heap: {:p}-{:p}, stack: {:p}-{:p}, used: {}, unused: {}) [\n",
            self.heap_start(),
            self.heap_top(),
            self.stack_start,
            self.stack_end,
            self.heap_used() + self.stack_used(),
            self.unused(),
        ))?;
        let mut pos = self.heap_start();
        while pos < self.heap_top() {
            unsafe {
                let term = &*pos;
                let skip = term.arity();
                f.write_fmt(format_args!(
                    "  {:p}: {:0bit_len$b} {:?}({:08x})\n",
                    pos,
                    *(pos as *const usize),
                    term.type_of(),
                    term.as_usize(),
                    bit_len = (bit_size_of::<usize>())
                ))?;
                pos = pos.add(1 + skip);
            }
        }
        f.write_str("  ==== END HEAP ====\n")?;
        f.write_str("  ==== START UNUSED ====\n")?;
        while pos < self.stack_start {
            unsafe {
                f.write_fmt(format_args!(
                    "  {:p}: {:0bit_len$b}\n",
                    pos,
                    *(pos as *const usize),
                    bit_len = (bit_size_of::<usize>())
                ))?;
                pos = pos.add(1);
            }
        }
        f.write_str("  ==== END UNUSED ====\n")?;
        f.write_str("  ==== START STACK ==== \n")?;
        pos = self.stack_start;
        while pos < self.stack_end {
            unsafe {
                let term = &*pos;

                let skip = term.arity();

                f.write_fmt(format_args!(
                    "  {:p}: {:0bit_len$b} {:?}\n",
                    pos,
                    *(pos as *const usize),
                    term,
                    bit_len = (bit_size_of::<usize>())
                ))?;
                pos = pos.add(1 + skip);
            }
        }
        f.write_str("]\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::TryInto;
    use core::ptr;

    use crate::erts;
    use crate::erts::process::alloc::{self, HeapAlloc};
    use crate::{atom, fixnum};

    #[test]
    fn young_heap_alloc() {
        let (heap, heap_size) = alloc::default_heap().unwrap();
        let mut yh = YoungHeap::new(heap, heap_size);

        assert_eq!(yh.heap_size(), heap_size);
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
        let num = fixnum!(101);
        let string = "test";
        let string_len = string.len();
        let string_term = yh.heapbin_from_str(string).unwrap();
        let string_term_size = Layout::for_value(string_term.as_ref()).size();
        let list = ListBuilder::new(&mut yh)
            .push(num)
            .push(string_term.into())
            .finish()
            .unwrap();
        let list_term: Term = list.into();
        let list_size =
            erts::to_word_size((mem::size_of::<Cons>() * 2) + string_term_size + string_len);
        assert_eq!(yh.heap_used(), list_size);
        assert!(list_term.is_non_empty_list());
        let list_ptr: *mut Cons = list_term.dyn_cast();
        let first_cell = unsafe { &*list_ptr };
        assert!(first_cell.head.is_smallint());
        assert!(first_cell.tail.is_non_empty_list());
        let tail_ptr: *mut Cons = first_cell.tail.dyn_cast();
        let second_cell = unsafe { &*tail_ptr };
        assert!(second_cell.head.is_boxed());
        let bin_ptr: *mut Term = second_cell.head.dyn_cast();
        let bin_term = unsafe { *bin_ptr };
        assert!(bin_term.is_heapbin());
        let hb = unsafe { HeapBin::from_raw_term(bin_ptr) };
        assert_eq!(hb.as_str(), string);
    }

    #[test]
    fn young_heap_stack_alloc() {
        let (heap, heap_size) = alloc::default_heap().unwrap();
        let mut yh = YoungHeap::new(heap, heap_size);

        // Allocate the list `[101, :foo]` on the stack
        let num = fixnum!(101);
        let foo = atom!("foo");
        let _list_term = HeaplessListBuilder::new(&mut yh)
            .push(num)
            .push(foo)
            .finish()
            .unwrap();
        // 2 cons cells + 1 box term
        let list_size = erts::to_word_size(mem::size_of::<Cons>() * 2) + 1;

        assert_eq!(yh.heap_used(), 0);
        assert_eq!(yh.stack_used(), list_size);
        assert_eq!(yh.stack_size(), 1);

        // Allocate a binary and put the pointer on the stack
        let string = "bar";
        let string_term = yh.heapbin_from_str(string).unwrap();
        unsafe {
            let ptr = yh.alloca(1).unwrap().as_ptr();
            ptr::write(ptr, string_term.into());
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
        assert_eq!(slot_term_addr, unsafe { yh.stack_start.add(1) });
        let slot_term = unsafe { *slot_term_addr };
        assert!(slot_term.is_non_empty_list());
        let list: Boxed<Cons> = slot_term.try_into().unwrap();
        assert!(list.head.is_smallint());
        assert!(list.tail.is_non_empty_list());
        let tail: Boxed<Cons> = list.tail.try_into().unwrap();
        assert!(tail.head.is_atom());
        assert!(tail.tail.is_nil());
    }
}

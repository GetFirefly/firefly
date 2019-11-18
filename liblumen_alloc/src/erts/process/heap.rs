use core::alloc::Layout;
use core::ptr::NonNull;

use log::trace;

use liblumen_core::util::pointer::distance_absolute;

use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::{Boxed, ProcBin, Term};

use super::alloc::{self, *};
use super::gc::{self, *};
use super::{Process, ProcessFlags};

/// This struct contains the actual semi-space heap that stack/heap allocations
/// are delegated to, and provides coordination for garbage collection of the
/// heap given the current process context.
#[derive(Debug)]
#[repr(C)]
pub struct ProcessHeap {
    // the number of minor collections
    pub(super) gen_gc_count: usize,
    // The semi-space generational heap
    heap: SemispaceProcessHeap,
}
impl ProcessHeap {
    pub fn new(heap: *mut Term, heap_size: usize) -> Self {
        let young = YoungHeap::new(heap, heap_size);
        let old = OldHeap::default();
        let heap = SemispaceHeap::new(young, old);
        Self {
            gen_gc_count: 0,
            heap,
        }
    }

    /// Returns true if this heap should be garbage collected
    #[inline]
    pub fn should_collect(&self, gc_threshold: f64) -> bool {
        self.heap.should_collect(gc_threshold)
    }

    #[cfg(test)]
    pub(super) fn heap(&self) -> &SemispaceProcessHeap {
        &self.heap
    }

    /// Runs garbage collection against the current heap
    ///
    /// This function and its helpers handle the following concerns:
    ///
    /// - Calculating the size of the new heap
    /// - Allocating a new heap
    /// - Setting any necessary process flags
    /// - Freeing dead heap fragments
    /// - Tracking statistics
    /// - Coordinating heap shrinkage/growth post-collection
    ///
    /// In other words, most of the process-specific parts occur in
    /// this module, while the more general garbage collection uses
    /// the GC infrastructure defined elsewhere.
    ///
    /// This separation makes it easier to test components of GC
    /// independently, and provides opportunity to inject specific
    /// components to test integrations of those components in isolation.
    #[inline]
    pub fn garbage_collect(
        &mut self,
        process: &Process,
        needed: usize,
        mut roots: RootSet,
    ) -> Result<usize, GcError> {
        // The primary source of roots we add is the process stack
        let young = self.heap.young_generation_mut();
        let sp = young.stack_pointer();
        let stack_size = young.stack_size();
        roots.push_range(sp, stack_size);

        // Initialize the collector
        // Determine if the current collection requires a full sweep or not
        if process.needs_fullsweep() || self.gen_gc_count >= process.max_gen_gcs {
            self.collect_full(process, needed, roots)
        } else {
            self.collect_minor(process, needed, roots)
        }
    }

    /// Handles the specific details required to initialize and execute a full sweep garbage
    /// collection
    fn collect_full(
        &mut self,
        process: &Process,
        needed: usize,
        roots: RootSet,
    ) -> Result<usize, GcError> {
        trace!("Performing a full sweep garbage collection");

        // Determine the estimated size for the new heap which will receive all live data
        let old_heap_size = self.heap.old_generation().heap_used();
        let young = self.heap.young_generation();
        let off_heap_size = process.off_heap_size();
        let size_before = young.heap_used() + old_heap_size + off_heap_size;

        // Conservatively pad out estimated size to include space for the number of words `needed`
        // free
        let baseline_estimate = young.stack_used() + size_before;
        let padded_estimate = baseline_estimate + needed;
        // If we already have a large enough heap, we don't need to grow it, but if the GROW flag is
        // set, then we should do it anyway, since it will prevent us from doing another full
        // collection for awhile (assuming one is not forced)
        let baseline_size = alloc::next_heap_size(padded_estimate);
        let new_heap_size =
            if baseline_size == young.heap_size() && process.should_force_heap_growth() {
                alloc::next_heap_size(baseline_size)
            } else {
                baseline_size
            };

        // Verify that our projected heap size is not going to blow the max heap size, if set
        // NOTE: When this happens, we will be left with no choice but to kill the process
        if process.max_heap_size > 0 && process.max_heap_size < new_heap_size {
            return Err(GcError::MaxHeapSizeExceeded);
        }

        // Unset heap_grow and need_fullsweep flags, because we are doing both
        process
            .flags
            .clear(ProcessFlags::GrowHeap | ProcessFlags::NeedFullSweep);

        // Allocate target heap (new young generation)
        let ptr = alloc::heap(new_heap_size).map_err(|alloc| GcError::Alloc(alloc))?;
        let mut target = YoungHeap::new(ptr, new_heap_size);

        // Initialize collector
        let _moved = {
            let gc_type = FullCollection::new(&mut self.heap, &mut target);
            let mut gc = ProcessCollector::new(roots, gc_type);
            // Run the collector
            gc.garbage_collect()?
        };

        // TODO: Move messages to be stored on-heap, on to the heap

        // Now that all live data has been swept on to the new heap, we can
        // clean up all of the off heap fragments that we still have laying around
        process.sweep_off_heap();

        // Reset the generational GC counter
        self.gen_gc_count = 0;

        // Calculate reclamation for tracing
        let young = self.heap.young_generation();
        let stack_used = young.stack_used();
        let heap_used = young.heap_used();
        let size_after = stack_used + heap_used + process.off_heap_size();
        if size_before >= size_after {
            trace!(
                "Full sweep reclaimed {} words of garbage",
                size_before - size_after
            );
        } else {
            trace!(
                "Full sweep resulted in heap growth of {} words",
                size_after - size_before
            );
        }

        // Determine if we oversized the heap and should shrink it
        //
        // This isn't strictly part of collection, but rather an
        // operation we require to ensure that we catch pathological
        // cases and handle them:
        //
        // - Unable to free enough space or allocate a large enough heap
        // - We over-allocated by a significant margin, and need to shrink
        // - We shrunk our heap usage, but not our heap, and need to shrink
        // - We under-allocated by a significant margin, and need to grow
        let needed_after = needed + stack_used + heap_used;
        let total_size = young.heap_size();

        // If this assertion fails, something went horribly wrong, our collection
        // resulted in a heap that was smaller than even our worst case estimate,
        // which means we almost certainly have a bug
        assert!(
            total_size >= needed_after,
            "completed GC (full), but the heap size needed ({}) exceeds even the most pessimistic estimate ({}), this must be a bug!",
            needed_after,
            total_size,
        );

        // Check if the needed space consumes more than 75% of the new heap,
        // and if so, schedule some heap growth to try and get ahead of allocations
        // failing due to lack of space
        if total_size * 3 < needed_after * 4 {
            process.flags.set(ProcessFlags::GrowHeap);
            return Ok(gc::estimate_cost(size_after, 0));
        }

        // Check if the needed space consumes less than 25% of the new heap,
        // and if so, shrink the new heap immediately to free the unused space
        if total_size > needed_after * 4 && process.min_heap_size < total_size {
            // Shrink to double our estimated need
            let mut estimate = needed_after * 2;
            // If our estimated need is too low, round up to the min heap size;
            // otherwise, calculate the next heap size bucket our need falls in
            if estimate < process.min_heap_size {
                estimate = process.min_heap_size;
            } else {
                estimate = alloc::next_heap_size(estimate);
            }

            // As a sanity check, only shrink the heap if the estimate is
            // actually smaller than the current heap size
            if estimate < total_size {
                self.shrink_young_heap(estimate);
                // The final cost of this GC needs to account for the moved heap
                Ok(gc::estimate_cost(size_after, size_after))
            } else {
                // We're not actually going to shrink, so our
                // cost is based purely on the size of the new heap
                Ok(gc::estimate_cost(size_after, 0))
            }
        } else {
            // No shrink required, so our cost is based purely on the size of the new heap
            Ok(gc::estimate_cost(size_after, 0))
        }
    }

    /// Handles the specific details required to initialize and execute a minor garbage collection
    fn collect_minor(
        &mut self,
        process: &Process,
        needed: usize,
        roots: RootSet,
    ) -> Result<usize, GcError> {
        trace!("Performing a minor garbage collection");

        // Determine the estimated size for the new heap which will receive immature live data
        let off_heap_size = process.off_heap_size();
        let young = self.heap.young_generation();
        let size_before = young.heap_used() + off_heap_size;
        let stack_size = young.stack_used();

        // Calculate mature region
        let mature_size = young.mature_size();

        // Verify that our projected heap size does not exceed
        // the max heap size, if one was configured.
        //
        // If a max heap size is set, make sure we're not going to exceed it
        if process.max_heap_size > 0 {
            // First, check if we have exceeded the max heap size
            let mut heap_size = size_before;
            // In this estimate, our stack size includes unused area between stack and heap
            let stack_size = stack_size + young.heap_available();
            let old = self.heap.old_generation();
            // Add potential old heap size
            if !old.active() && mature_size > 0 {
                heap_size += alloc::next_heap_size(size_before);
            } else if old.active() {
                heap_size += old.heap_used();
            }

            // Add potential new young heap size, conservatively estimating
            // the worst case scenario where we free no memory and need to
            // reclaim `needed` words. We grow the projected size until there
            // is at least enough memory for the current heap + `needed`
            let baseline_size = stack_size + size_before + needed;
            heap_size += alloc::next_heap_size(baseline_size);

            // When this error type is returned, a full sweep will be triggered
            if heap_size > process.max_heap_size {
                return Err(GcError::MaxHeapSizeExceeded);
            }
        }

        // Allocate an old heap if we don't have one and one is needed
        if !self.heap.old_generation().active() && mature_size > 0 {
            let size = alloc::next_heap_size(size_before);
            let ptr = alloc::heap(size).map_err(|alloc| GcError::Alloc(alloc))?;
            let _ = self.heap.swap_old(OldHeap::new(ptr, size));
        }

        // If the old heap isn't present, or isn't large enough to hold the
        // mature objects in the young generation, then a full sweep is required
        let old = self.heap.old_generation();
        if old.active() && mature_size > old.heap_available() {
            return Err(GcError::FullsweepRequired);
        }

        let prev_old_top = old.heap_top();
        let baseline_size = stack_size + size_before + needed;
        // While we expect that we will free memory during collection,
        // we want to avoid the case where we collect and then find that
        // the new heap is too small to meet the need that triggered the
        // collection in the first place. Better to shrink it post-collection
        // than to require growing it and re-updating all the roots again
        let new_size = alloc::next_heap_size(baseline_size);

        // Allocate new young generation heap
        let ptr = alloc::heap(new_size).map_err(|alloc| GcError::Alloc(alloc))?;
        let new_young = YoungHeap::new(ptr, new_size);

        // Swap it with the existing young generation heap
        let mut source = self.heap.swap_young(new_young);

        let _moved = {
            // Initialize the collector to collect objects into the new semi-space heap
            let gc_type = MinorCollection::new(&mut source, &mut self.heap);
            let mut gc = ProcessCollector::new(roots, gc_type);
            // Run the collector
            gc.garbage_collect()?
        };

        // Now that all live data has been swept on to the new heap, we can
        // clean up all of the off heap fragments that we still have laying around
        process.sweep_off_heap();

        // Increment the generational GC counter
        self.gen_gc_count += 1;

        // TODO: if using on-heap messages, move messages in the queue to the heap

        // Calculate memory usage after collection
        let old = self.heap.old_generation();
        let young = self.heap.young_generation();
        let new_mature_size = distance_absolute(old.heap_top(), prev_old_top);
        let heap_used = young.heap_used();
        let size_after = new_mature_size + heap_used; // TODO: add process.mbuf_size
        let needed_after = heap_used + needed + stack_size;

        // Excessively large heaps should be shrunk, but don't even bother on reasonable small heaps
        //
        // The reason for this is that after tenuring, we often use a really small portion of the
        // new heap, therefore unless the heap size is substantial, we don't want to shrink
        let heap_size = young.heap_size();
        let is_oversized = heap_size > needed_after * 4;
        let old_heap_size = old.heap_size();
        let should_shrink = is_oversized && (heap_size > 8000 || heap_size > old_heap_size);
        if should_shrink {
            // We are going to shrink the heap to 3x the size of our current need,
            // at this point we already know that the heap is more than 4x our current need,
            // so this provides a reasonable baseline heap usage of 33%
            let mut estimate = needed_after * 3;
            // However, if this estimate is very small compared to the size of
            // the old generation, then we are likely going to be forced to reallocate
            // sooner rather than later, as the old generation would seem to indicate
            // that we allocate many objects.
            //
            // We determine our estimate to be too small if it is less than 10% the
            // size of the old generation. In this situation, we set our estimate to
            // be 25% of the old generation heap size
            if estimate * 9 < old_heap_size {
                estimate = old_heap_size / 8;
            }

            // If the new estimate is less than the min heap size, then round up;
            // otherwise, round the estimate up to the nearest heap size bucket
            if estimate < process.min_heap_size {
                estimate = process.min_heap_size;
            } else {
                estimate = alloc::next_heap_size(estimate);
            }

            // As a sanity check, only shrink if our revised estimate is
            // actually smaller than the current heap size
            if estimate < heap_size {
                self.shrink_young_heap(estimate);
                // Our final cost should account for the moved heap
                Ok(gc::estimate_cost(size_after, heap_used))
            } else {
                // We're not actually going to shrink, so our cost
                // is entirely based on the size of the new heap
                Ok(gc::estimate_cost(size_after, 0))
            }
        } else {
            // No shrink required, so our cost is based
            // on the size of the new heap only
            Ok(gc::estimate_cost(size_after, 0))
        }
    }

    /// In some cases, after a minor collection we may find that we have over-allocated for the
    /// new young heap, this is because we make a conservative estimate as to how much space will
    /// be needed, and if our collections are effective, that may leave a lot of unused space.
    ///
    /// We only really care about excessively large heaps, as reasonable heap sizes, even if
    /// larger than needed, are not worth the effort to shrink. In the case of the exessively
    /// large heaps though, we need to shift the stack from the end of the heap to its new
    /// position, and then reallocate (in place) the memory underlying the heap.
    ///
    /// Since we control the allocator for process heaps, it is not necessary for us to handle
    /// the case of trying to reallocate the heap and having it move on us, beyond asserting
    /// that the heap is not moved. In BEAM, they have to account for that condition, as the
    /// allocators do not provide a `realloc_in_place` API
    fn shrink_young_heap(&mut self, new_size: usize) {
        unsafe { self.heap.young_generation_mut().shrink(new_size) }
    }
}
impl HeapAlloc for ProcessHeap {
    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        self.heap.alloc_layout(layout)
    }
}
impl Heap for ProcessHeap {
    #[inline]
    fn heap_start(&self) -> *mut Term {
        self.heap.heap_start()
    }

    #[inline]
    fn heap_top(&self) -> *mut Term {
        self.heap.heap_top()
    }

    #[inline]
    fn heap_end(&self) -> *mut Term {
        self.heap.heap_end()
    }

    #[inline]
    fn heap_size(&self) -> usize {
        self.heap.heap_size()
    }

    #[inline]
    fn heap_used(&self) -> usize {
        self.heap.heap_used()
    }

    #[inline]
    fn heap_available(&self) -> usize {
        self.heap.heap_available()
    }

    #[inline]
    fn contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.heap.contains(ptr)
    }

    #[inline]
    fn is_owner<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.heap.contains(ptr)
    }
}
impl VirtualHeap<ProcBin> for ProcessHeap {
    #[inline]
    fn virtual_size(&self) -> usize {
        self.heap.virtual_size()
    }

    #[inline]
    fn virtual_heap_used(&self) -> usize {
        self.heap.virtual_heap_used()
    }

    #[inline]
    fn virtual_heap_unused(&self) -> usize {
        self.heap.virtual_heap_unused()
    }
}
impl VirtualAllocator<ProcBin> for ProcessHeap {
    #[inline]
    fn virtual_alloc(&mut self, value: Boxed<ProcBin>) {
        self.heap.virtual_alloc(value)
    }

    #[inline]
    fn virtual_free(&mut self, value: Boxed<ProcBin>) {
        self.heap.virtual_free(value);
    }

    #[inline]
    fn virtual_unlink(&mut self, value: Boxed<ProcBin>) {
        self.heap.virtual_unlink(value);
    }

    #[inline]
    fn virtual_pop(&mut self, value: Boxed<ProcBin>) -> ProcBin {
        self.heap.virtual_pop(value)
    }

    #[inline]
    fn virtual_contains<P: ?Sized>(&self, value: *const P) -> bool {
        self.heap.virtual_contains(value)
    }

    #[inline]
    unsafe fn virtual_clear(&mut self) {
        self.heap.virtual_clear();
    }
}
impl StackAlloc for ProcessHeap {
    #[inline]
    unsafe fn alloca(&mut self, need: usize) -> AllocResult<NonNull<Term>> {
        self.heap.alloca(need)
    }

    #[inline]
    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        self.heap.alloca_unchecked(need)
    }
}
impl StackPrimitives for ProcessHeap {
    #[inline]
    fn stack_size(&self) -> usize {
        self.heap.stack_size()
    }

    #[inline]
    unsafe fn set_stack_size(&mut self, size: usize) {
        self.heap.set_stack_size(size);
    }

    #[inline]
    fn stack_pointer(&mut self) -> *mut Term {
        self.heap.stack_pointer()
    }

    #[inline]
    unsafe fn set_stack_pointer(&mut self, sp: *mut Term) {
        self.heap.set_stack_pointer(sp);
    }

    #[inline]
    fn stack_used(&self) -> usize {
        self.heap.stack_used()
    }

    #[inline]
    fn stack_available(&self) -> usize {
        self.heap.stack_available()
    }

    #[inline]
    fn stack_slot(&mut self, n: usize) -> Option<Term> {
        self.heap.stack_slot(n)
    }

    #[inline]
    fn stack_popn(&mut self, n: usize) {
        self.heap.stack_popn(n);
    }
}

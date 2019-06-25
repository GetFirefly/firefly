use core::alloc::AllocErr;
use core::ptr;

use intrusive_collections::UnsafeRef;
use log::trace;

use liblumen_core::util::pointer::{distance_absolute, in_area};

use super::*;
use crate::erts::process::alloc;
use crate::erts::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum CollectionType {
    Minor,
    Full,
}

/// This type implements the logic for performing a single garbage
/// collection on a process, given a set of roots from which live
/// values can be derived.
///
/// Each time a collection is required, an instance of this collector
/// should be created, with a reference to the process to collect, and
/// a set of roots from which live values will be traced. Once created,
/// the collector only runs when `collect` is called, and while it can
/// be safely called multiple times, the collector itself cannot be
/// held long-term, as it requires a mutable reference to the parent
/// process, and it is not necessary to keep alive, nor expensive to
/// recreate on-demand.
///
/// This collector can be described as having all of the following properties:
///
/// - It is a precise/tracing collector; the algorithm works by working from a set of "root" values
///   which are live somewhere, and which point into either the young or old heaps, or the process
///   stack
/// - It is generational; values that survive two minor collections are moved into the old
///   generation on the second collection, this is facilitated by the use of a high water mark in
///   the young heap which indicates where the last collection finished, everything under that point
///   which is still live is tenured
/// - It is a semi-space collector; minor collections consist of allocating a new young generation
///   and moving live values to the new heap, and if applicable, moving tenured values to the old
///   generation.
/// - It has some mark/sweep heritage; while the primary driver of liveness is the root set, when
///   all root set referenced values have been moved to the new heap, the new heap is traversed
///   word-by-word to locate any values which still point into the old young generation heap which
///   is about to be discarded. This is somewhat similar to a mark and sweep pass, except we are not
///   marking any values and then cleaning them up later, instead we move them immediately, almost
///   exactly the same way that we work with the root set.
///
/// The design is heavily inspired by that of the BEAM garbage collector,
/// with only minor changes. The current version today is not fully equal
/// to its cousin, as it doesn't handle the full set of capabilities the
/// BEAM supports, but as Lumen builds up its feature set, we will add those
/// missing features here correspondingly
pub struct GarbageCollector<'p> {
    mode: CollectionType,
    process: &'p mut ProcessControlBlock,
    roots: RootSet,
}
impl<'p> GarbageCollector<'p> {
    /// Initializes the collector with the given process and root set
    pub fn new(process: &'p mut ProcessControlBlock, roots: RootSet) -> Self {
        let mode = if Self::need_fullsweep(process) {
            CollectionType::Full
        } else {
            CollectionType::Minor
        };
        Self {
            mode,
            process,
            roots,
        }
    }

    /// Invokes the collector and uses the provided `need` (in words)
    /// to determine whether or not collection was successful/aggressive enough
    #[inline]
    pub fn collect(&mut self, need: usize) -> Result<usize, GcError> {
        match self.mode {
            CollectionType::Minor => self.minor_sweep(need),
            CollectionType::Full => self.full_sweep(need),
        }
    }

    //
    // - Allocate to space
    // - Scan roots to find heap objects in from space to copy
    //      - Place move marker in previous heap position pointing to new heap position
    // - Scan to-space for pointers to from-space, and copy objects to to-space
    //      - NOTE: skip non-term values found on heap
    //      - Values copied here are allocated after `scan_stop` line, then `scan_stop` is adjusted
    //        past the object
    // - When `scan_stop` hits `scan_start`, we're done with the minor collection
    // - Deallocate from space
    fn full_sweep(&mut self, need: usize) -> Result<usize, GcError> {
        trace!("Performing a full sweep garbage collection");

        let old_heap_size = self.process.old.heap_used();

        // Determine the estimated size for the new heap which will receive all live data
        let stack_size = self.process.young.stack_used();
        let off_heap_size = self.process.off_heap_size();
        let size_before = self.process.young.heap_used() + old_heap_size + off_heap_size;
        // Conservatively pad out estimated size to include space for the requested `need`
        let mut new_size = alloc::next_heap_size(stack_size + size_before);
        while new_size < (need + stack_size + size_before) {
            new_size = alloc::next_heap_size(new_size);
        }
        // If we already have a large enough heap, we don't need to grow it, but if the GROW flag is
        // set, then we should do it anyway, since it will prevent us from doing another full
        // collection for awhile (assuming one is not forced)
        if new_size == self.process.young.size() && self.should_force_heap_growth() {
            new_size = alloc::next_heap_size(new_size);
        }
        // Verify that our projected heap size is not going to blow the max heap size, if set
        // NOTE: When this happens, we will be left with no choice but to kill the process
        if self.process.max_heap_size > 0 && self.process.max_heap_size < new_size {
            return Err(GcError::MaxHeapSizeExceeded);
        }
        // Unset heap_grow and need_fullsweep flags, because we are doing both
        self.process
            .flags
            .clear(ProcessFlag::GrowHeap | ProcessFlag::NeedFullSweep);
        // Allocate new heap
        let new_heap_start = alloc::heap(new_size).map_err(|_| GcError::AllocErr)?;
        let mut new_heap = YoungHeap::new(new_heap_start, new_size);
        // Follow roots and copy values to appropriate heaps
        unsafe {
            for root in self.roots.iter() {
                let root: *mut Term = *(root as *const _);
                let term = *root;
                if term.is_boxed() {
                    let ptr = term.boxed_val();
                    let boxed = *ptr;
                    if is_move_marker(boxed) {
                        // Replace the boxed move marker with the "real" box
                        assert!(boxed.is_boxed());
                        ptr::write(root, boxed);
                    } else if !term.is_literal() {
                        if boxed.is_procbin() {
                            // First we need to remove the procbin from its old virtual heap
                            let old_bin = &*(ptr as *mut ProcBin);
                            if self.process.young.virtual_heap_contains(old_bin) {
                                self.process.young.virtual_heap_unlink(old_bin);
                            } else {
                                self.process.old.virtual_heap_unlink(old_bin);
                            }
                            // Move to top of the new heap
                            new_heap.move_into(root, ptr, boxed);
                            // Then add the procbin to the new virtual heap
                            let marker = *ptr;
                            assert!(marker.is_boxed());
                            let bin_ptr = marker.boxed_val() as *mut ProcBin;
                            let bin = &*bin_ptr;
                            new_heap.virtual_alloc(bin);
                        } else {
                            // Move into new heap
                            new_heap.move_into(root, ptr, boxed);
                        }
                    }
                } else if term.is_list() {
                    let ptr = term.list_val();
                    let cons = *ptr;
                    if cons.is_move_marker() {
                        // Replace the move marker with the "real" value
                        ptr::write(root, cons.tail);
                    } else if !term.is_literal() {
                        // Move into new heap
                        new_heap.move_cons_into(root, ptr, cons);
                    }
                }
            }
        }
        // All references in the roots point to the new heap, but most of
        // the references in the values we just moved still point back to
        // the old heaps
        new_heap.full_sweep(&mut self.process.young, &mut self.process.old);

        // Now that all live data has been swept on to the new heap, we can
        // clean up all of the off heap fragments that we still have laying around
        self.sweep_off_heap();

        // Free the old generation heap, as it is no longer used post-sweep
        if self.process.old.active() {
            let old_heap_start = self.process.old.heap_start();
            let old_heap_size = self.process.old.size();
            unsafe { alloc::free(old_heap_start, old_heap_size) };
            self.process.old = OldHeap::empty();
        }

        // Move the stack to the end of the new heap
        let old_stack_start = self.process.young.stack_pointer();
        let old_stack_slots = self.process.young.stack_size();
        unsafe {
            let new_stack_start = new_heap.alloca_unchecked(stack_size).as_ptr();
            ptr::copy_nonoverlapping(old_stack_start, new_stack_start, stack_size);
            new_heap.set_stack_pointer(new_stack_start);
            new_heap.set_stack_size(old_stack_slots);
        }

        // Free the old young generation heap, as it is no longer used now that the stack is moved
        let young_heap_start = self.process.young.heap_start();
        let young_heap_size = self.process.young.size();
        unsafe { alloc::free(young_heap_start, young_heap_size) };

        // Update the process to use the new heap
        new_heap.set_high_water_mark();
        self.process.young = new_heap;
        self.process.gen_gc_count = 0;

        // TODO: Move messages to be stored on-heap, on to the heap
        // Check invariants
        self.sanity_check();

        // Calculate reclamation for tracing
        let stack_used = self.process.young.stack_used();
        let heap_used = self.process.young.heap_used();
        let size_after = heap_used + stack_used + self.process.off_heap_size();
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
        let mut adjusted = false;
        let need_after = need + stack_used + heap_used;
        let total_size = self.process.young.size();
        if total_size < need_after {
            // Something went horribly wrong, our collection resulted in a heap
            // that was smaller than even our worst case estimate, which means
            // we almost certainly have a bug
            panic!("Full sweep finished, but the needed size exceeds even the most pessimistic estimate, this must be a bug");
        } else if total_size * 3 < need_after * 4 {
            // `need_after` requires more than 75% of the current size, schedule some growth
            self.process.flags.set(ProcessFlag::GrowHeap);
        } else if total_size > need_after * 4 && self.process.min_heap_size < total_size {
            // We need less than 25% of the current heap, shrink
            let wanted = need_after * 2;
            let size = if wanted < self.process.min_heap_size {
                self.process.min_heap_size
            } else {
                alloc::next_heap_size(wanted)
            };
            if size < total_size {
                self.shrink_young_heap(size);
                adjusted = true;
            }
        }
        // Calculate the cost of the collection
        if adjusted {
            Ok(Self::gc_cost(size_after, size_after))
        } else {
            Ok(Self::gc_cost(size_after, 0))
        }
    }

    /// This function is the main entry point for a minor collection
    ///
    /// 1. Verify that we are not going to exceed the maximum heap size
    fn minor_sweep(&mut self, need: usize) -> Result<usize, GcError> {
        trace!("Performing a minor garbage collection");

        let off_heap_size = self.process.off_heap_size();
        let size_before = self.process.young.heap_used() + off_heap_size;
        let mature_size = self.process.young.mature_size();

        // Verify that our projected heap size does not exceed
        // the max heap size, if one was configured
        self.verify_heap_size(need, size_before, mature_size)?;

        // Allocate an old heap if we don't have one and one is needed
        self.ensure_old_heap(size_before, mature_size)
            .map_err(|_| GcError::AllocErr)?;

        // Do a minor collection if there is an old heap and it is large enough
        if self.process.old.active() && mature_size > self.process.old.heap_available() {
            return Err(GcError::FullsweepRequired);
        }

        let prev_old_top = self.process.old.heap_pointer();
        let stack_size = self.process.young.stack_used();
        let mut new_size = alloc::next_heap_size(stack_size + size_before);
        while new_size < (stack_size + size_before + need) {
            // While we expect that we will free memory during collection,
            // we want to avoid the case where we collect and then find that
            // the new heap is too small to meet the need that triggered the
            // collection in the first place. Better to shrink it post-collection
            // than to require growing it and re-updating all the roots again
            new_size = alloc::next_heap_size(new_size);
        }

        // Perform the "meat" of the minor collection
        unsafe { self.do_minor_sweep(self.process.young.heap_start(), mature_size, new_size)? };

        // TODO: if using on-heap messages, move messages in the queue to the heap

        let new_mature = distance_absolute(self.process.old.heap_pointer(), prev_old_top);
        let heap_used = self.process.young.heap_used();
        let size_after = new_mature + heap_used; // + process.mbuf_size

        self.sanity_check();

        self.process.gen_gc_count += 1;
        let need_after = heap_used + need + stack_size;

        // Excessively large heaps should be shrunk, but don't even bother on reasonable small heaps
        //
        // The reason for this is that after tenuring, we often use a really small portion of the
        // new heap, therefore unless the heap size is substantial, we don't want to shrink
        let mut adjust_size = 0;
        let oversized_heap = heap_used > need_after * 4;
        let old_heap_size = self.process.old.size();
        let shrink = oversized_heap && (heap_used > 8000 || heap_used > old_heap_size);
        if shrink {
            let mut wanted = need_after * 3;
            // Additional test to make sure we don't make the heap too small
            // compared to the size of the older generation heap
            if wanted * 9 < old_heap_size {
                let new_wanted = old_heap_size / 8;
                if new_wanted > wanted {
                    wanted = new_wanted;
                }
            }

            wanted = if wanted < self.process.min_heap_size {
                self.process.min_heap_size
            } else {
                alloc::next_heap_size(wanted)
            };

            if wanted < heap_used {
                self.shrink_young_heap(wanted);
                adjust_size = heap_used;
            }
        }

        Ok(Self::gc_cost(size_after, adjust_size))
    }

    unsafe fn do_minor_sweep(
        &mut self,
        mature: *mut Term,
        mature_size: usize,
        new_size: usize,
    ) -> Result<(), GcError> {
        let old_top = self.process.old.heap_pointer();
        let mature_end = mature.offset(mature_size as isize);

        // Allocate new tospace (young generation)
        let new_young_start = alloc::heap(new_size).map_err(|_| GcError::AllocErr)?;
        let mut new_young = YoungHeap::new(new_young_start, new_size);

        // Follow roots and copy values to appropriate heaps
        for root in self.roots.iter() {
            let root: *mut Term = *(root as *const _);
            let term = *root;
            if term.is_boxed() {
                let ptr = term.boxed_val();
                let boxed = *ptr;
                if is_move_marker(boxed) {
                    // Replace the boxed move marker with the "real" box
                    assert!(boxed.is_boxed());
                    ptr::write(root, boxed);
                } else if in_area(ptr, mature, mature_end) {
                    // The boxed value should be moved to the old generation,
                    // since the value is below the high water mark
                    if boxed.is_procbin() {
                        // First we need to remove the procbin from its old virtual heap
                        let old_bin = &*(ptr as *mut ProcBin);
                        self.process.young.virtual_heap_unlink(old_bin);
                        // Move to top of the old gen heap
                        self.process.old.move_into(root, ptr, boxed);
                        // Then add the procbin to the new virtual heap
                        let marker = *ptr;
                        assert!(marker.is_boxed());
                        let bin_ptr = marker.boxed_val() as *mut ProcBin;
                        let bin = &*bin_ptr;
                        self.process.old.virtual_alloc(bin);
                    } else {
                        self.process.old.move_into(root, ptr, boxed);
                    }
                } else if !term.is_literal() && !self.process.old.contains(ptr) {
                    // The boxed value is in the young generation
                    if boxed.is_procbin() {
                        // First we need to remove the procbin from its old virtual heap
                        let old_bin = &*(ptr as *mut ProcBin);
                        self.process.young.virtual_heap_unlink(old_bin);
                        // Move to top of the new heap
                        new_young.move_into(root, ptr, boxed);
                        // Then add the procbin to the new virtual heap
                        let marker = *ptr;
                        assert!(marker.is_boxed());
                        let bin_ptr = marker.boxed_val() as *mut ProcBin;
                        let bin = &*bin_ptr;
                        new_young.virtual_alloc(bin);
                    } else {
                        // Just a normal move into the new heap
                        new_young.move_into(root, ptr, boxed);
                    }
                }
            } else if term.is_list() {
                let ptr = term.list_val();
                let cons = *ptr;
                if cons.is_move_marker() {
                    // Replace the move marker with the "real" value
                    ptr::write(root, cons.tail);
                } else if in_area(ptr, mature, mature_end) {
                    // Move to old generation
                    self.process.old.move_cons_into(root, ptr, cons);
                } else if !term.is_literal() && !self.process.old.contains(ptr) {
                    // Move to new young heap
                    new_young.move_cons_into(root, ptr, cons);
                }
            }
        }

        // Now all references in the root set point to the new heap. However,
        // most references on the new heap point to the old heap so the next stage
        // is to scan through the new heap, evacuating data to the new heap until all
        // live references have been moved
        new_young.sweep(
            &mut self.process.young,
            &mut self.process.old,
            mature,
            mature_end,
        );

        // If we have been tenuring (we have an old generation and have moved values into it),
        // then those newly tenured values may hold references into the old young generation
        // heap, which is about to be freed, so we need to move them into the old generation
        // as well (we never allow pointers into the young generation from the old)
        if self.process.old.heap_pointer() as usize > old_top as usize {
            self.process.old.sweep(&mut new_young);
        }

        // Mark where this collection ended in the new heap
        new_young.set_high_water_mark();

        // Sweep off-heap fragments
        self.sweep_off_heap();

        // Copy stack to end of new heap
        new_young.copy_stack_from(&self.process.young);

        // Replace the now freed young heap with the one we've been building
        // NOTE: Because YoungHeap implements Drop, the old young heap will be freed correctly here
        self.process.young = new_young;

        Ok(())
    }

    fn sweep_off_heap(&mut self) {
        // We can drop all fragments, as the data they contain has either been moved
        // after the sweep that came just before this, so no other references
        // are possible, or it is garbage that can be collected
        //
        // When we drop the `HeapFragment`, its `Drop` implementation executes the
        // destructor of the term stored in the fragment, and then frees the memory
        // backing it. Since this takes care of any potential clean up we may need
        // to do automatically, we don't have to do any more than that here, at least
        // for now. In the future we may need to have more control over this, but
        // not in the current state of the system
        let mut off_heap = self.process.off_heap.lock();
        let mut cursor = off_heap.front_mut();
        while let Some(fragment_ref) = cursor.remove() {
            let fragment_ptr = UnsafeRef::into_raw(fragment_ref);
            unsafe { ptr::drop_in_place(fragment_ptr) };
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
        self.process.young.shrink(new_size)
    }

    /// Ensures the old heap is initialized, if required
    #[inline]
    fn ensure_old_heap(&mut self, size_before: usize, mature_size: usize) -> Result<(), AllocErr> {
        if !self.process.old.active() && mature_size > 0 {
            let size = alloc::next_heap_size(size_before);
            let start = alloc::heap(size)?;
            self.process.old = OldHeap::new(start, size);
        }
        Ok(())
    }

    /// Ensures the projected heap size does not exceed the maximum heap size configured
    #[inline]
    fn verify_heap_size(
        &self,
        need: usize,
        size_before: usize,
        mature_size: usize,
    ) -> Result<(), GcError> {
        let young = &self.process.young;
        let old = &self.process.old;

        // If a max heap size is set, make sure we're not going to exceed it
        if self.process.max_heap_size > 0 {
            // First, check if we have exceeded the max heap size
            let mut heap_size = size_before;
            // Includes unused area between stack and heap
            let stack_size = young.stack_used() + young.unused();
            // Add potential old heap size
            if !old.active() && mature_size > 0 {
                heap_size += alloc::next_heap_size(size_before);
            } else if old.active() {
                heap_size += old.heap_used();
            }
            // Add potential new young heap size, conservatively estimating
            // the worst case scenario where we free no memory and need to
            // reclaim `need` words. We grow the projected size until there
            // is at least enough memory for the current heap + `need`
            let mut new_heap_size = alloc::next_heap_size(stack_size + size_before);
            while new_heap_size < (need + stack_size + size_before) {
                new_heap_size = alloc::next_heap_size(new_heap_size);
            }
            heap_size += new_heap_size;

            // When this error type is returned, a full sweep will be triggered
            if heap_size > self.process.max_heap_size {
                return Err(GcError::MaxHeapSizeExceeded);
            }
        }

        Ok(())
    }

    /// Determines if the current collection requires a full sweep or not
    #[inline]
    fn need_fullsweep(process: &ProcessControlBlock) -> bool {
        if process.needs_fullsweep() {
            return true;
        }
        process.gen_gc_count >= process.max_gen_gcs
    }

    /// Determines if we should try and grow the heap even when not necessary
    #[inline]
    fn should_force_heap_growth(&self) -> bool {
        self.process.flags.is_set(ProcessFlag::GrowHeap)
    }

    /// Calculates the reduction count cost of a collection
    #[inline]
    fn gc_cost(moved_live_words: usize, resize_moved_words: usize) -> usize {
        let reds = (moved_live_words / 10) + (resize_moved_words / 100);
        if reds < 1 {
            1
        } else {
            reds
        }
    }

    /// Runs verification of heap invariants
    #[inline]
    fn sanity_check(&self) {
        self.process.young.sanity_check()
    }
}

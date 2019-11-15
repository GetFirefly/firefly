use core::mem;

use crate::erts::process::gc::{CollectionType, GcError, RootSet, OldHeap};
use crate::erts::process::gc::{FullSweep, MinorSweep, ReferenceCollection};
use crate::erts::process::alloc::{Heap, GenerationalHeap, VirtualAlloc};

use super::GarbageCollector;

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
///
/// NOTE: The portions of the collection process that are specific to the Process itself,
/// are handled by the `ProcessHeap::garbage_collect` function; this includes things like
/// cleaning up dead heap fragments, setting process flags for various conditions, calculating
/// statistics, and allocating/resizing heaps that the collector operates on. This keeps the
/// core collector general enough to be easily tested in isolation, and then plugged back in to
/// a `ProcessHeap` to test the higher-level integration.
pub struct ProcessCollector<T: CollectionType> {
    roots: RootSet,
    gc: T,
    moved: usize,
}
impl<G, T> ProcessCollector<G>
where
    T: Heap + VirtualAlloc,
    G: CollectionType<Target=T>,
{
    pub fn new(roots: RootSet, gc: G) -> Self {
        Self {
            roots,
            gc,
            moved: 0,
        }
    }

    /// Runs verification of heap invariants
    #[inline]
    fn sanity_check(&self) {
        self.gc.target().sanity_check()
    }
}
impl<'h> GarbageCollector<FullSweep<'h>> for ProcessCollector<FullSweep<'h>> {
    /// Invokes the collector and uses the provided `need` (in words)
    /// to determine whether or not collection was successful/aggressive enough
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
    #[inline]
    fn garbage_collect(&mut self) -> Result<usize, GcError> {
        use crate::erts::process::gc::collection_type::sweep_root;

        // Follow roots and copy values to appropriate heaps
        for mut root in self.roots.iter().copied() {
            let moved = unsafe { sweep_root(&mut self.gc, root.as_mut()) };
            self.moved += moved;
        }

        // Now that the new heap is seeded with roots, we can perform a sweep,
        // which examines the new heap for references in the old heaps and moves
        // them into the new
        self.moved += self.gc.collect();

        // Free the old generation heap, by swapping it out with a new empty
        // old heap, resulting in it being dropped. The old generation is no
        // longer used post-sweep, until the next minor collection with a mature
        // region occurs
        let source = self.gc.source_mut();
        source.swap_old(OldHeap::empty());

        // Move the stack to the end of the new heap
        let young = source.young_generation();
        let target = self.gc.target_mut();
        unsafe {
            target.copy_stack_from(young);
        }

        // Swap the target heap with the source young heap,
        // making the target the active heap
        //
        // A few things to note:
        //
        // - We use `swap` here because the new heap was initialized
        // outside of the collector, and when the collector returns,
        // that heap will be dropped when it falls out of scope. By
        // using `swap`, we are replacing the object that gets dropped
        // with the old young heap
        // - This invalidates the `target` reference, a new one must
        // be created if needed
        // - The reference given by `source.young_generation` is the
        // new heap that was `target` prior to this swap
        mem::swap(source.young_generation_mut(), target);

        let source = self.gc.source_mut();
        let young = source.young_generation_mut();

        // Reset the high water mark
        young.set_high_water_mark();

        // Check invariants
        self.sanity_check();

        Ok(self.moved)
    }
}


impl<'h> GarbageCollector<MinorSweep<'h>> for ProcessCollector<MinorSweep<'h>> {
    /// Invokes the collector and uses the provided `need` (in words)
    /// to determine whether or not collection was successful/aggressive enough
    ///
    /// This function is the main entry point for a minor collection
    ///
    /// 1. Verify that we are not going to exceed the maximum heap size
    #[inline]
    fn garbage_collect(&mut self) -> Result<usize, GcError> {
        use crate::erts::process::gc::collection_type::sweep_term;

        // Track the top of the old generation to see if we promote any mature objects
        let old_top = self.gc.target().old_generation().heap_top();

        // Follow roots and copy values to appropriate heaps
        for mut root in self.roots.iter().copied() {
            let moved = unsafe { sweep_term(&mut self.gc, root.as_mut()) };
            self.moved += moved;
        }

        // Now that the new heap is seeded with roots, we can perform a sweep,
        // which examines the new heap for references in the old heaps and moves
        // them into the new
        self.moved += self.gc.collect();

        // Get mutable references to both generations
        let target = self.gc.target_mut();
        let old = target.old_generation_mut();

        // If we have been tenuring (we have an old generation and have moved values into it),
        // then those newly tenured values may hold references into the old young generation
        // heap, which is about to be freed, so we need to move them into the old generation
        // as well (we never allow pointers into the young generation from the old)
        let has_tenured = old.heap_top() > old_top;
        if has_tenured {
            let mut rc = ReferenceCollection::new(self.gc.source_mut(), old);
            self.moved += rc.collect();
        }

        // Mark where this collection ended in the new heap
        let young = target.young_generation_mut();
        young.set_high_water_mark();

        // Copy stack to end of new heap
        unsafe { young.copy_stack_from(self.gc.source()); }

        // Check invariants
        self.sanity_check();

        Ok(self.moved)
    }
}

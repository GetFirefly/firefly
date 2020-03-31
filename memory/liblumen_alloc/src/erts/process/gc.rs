mod collection_type;
pub mod collector;
mod old_heap;
mod rootset;
mod sweep;
mod young_heap;

#[cfg(test)]
mod tests;

pub use self::collection_type::{
    CollectionType, FullCollection, MinorCollection, ReferenceCollection,
};
pub use self::collector::{GarbageCollector, ProcessCollector, SimpleCollector};
pub use self::old_heap::OldHeap;
pub use self::rootset::RootSet;
pub use self::sweep::{Sweep, Sweepable, Sweeper};
pub use self::young_heap::YoungHeap;

use super::alloc::SemispaceHeap;
use crate::erts::exception;
use thiserror::Error;

/// Represents the types of errors that can occur during garbage collection.
///
/// See the documentation for each variant to get general advice for how to
/// handle these errors
#[derive(Error, Debug, PartialEq, Eq)]
pub enum GcError {
    /// The system is out of memory, and there is not much you can do
    /// but panic, however this choice is left up to the caller
    #[error("unable to allocate memory for garbage collection")]
    Alloc(#[from] exception::Alloc),
    /// Occurs when a process is configured with a maximum heap size,
    /// and a projected heap growth is found to exceed the limit. In
    /// this situation the only meaningful thing to do is to kill the
    /// process
    #[error("maximum heap size exceeded")]
    MaxHeapSizeExceeded,
    /// Indicates that an allocation could not be filled without first
    /// performing a full sweep collection
    #[error("a full garbage collection sweep is required")]
    FullsweepRequired,
}

/// An enumeration of the generation types that can be targeted
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Generation {
    Young,
    Old,
}

/// A type alias for the semi-space heap which is wrapped by `ProcessHeap`
pub type SemispaceProcessHeap = SemispaceHeap<YoungHeap, OldHeap>;

/// A type alias for the type of a full collection which operates on the standard
/// process heap configuration
pub type FullSweep<'a> = FullCollection<'a, SemispaceProcessHeap, YoungHeap>;
/// A type alias for the type of a minor collection which operates on the standard
/// process heap configuration
pub type MinorSweep<'a> = MinorCollection<'a, YoungHeap, SemispaceProcessHeap>;
/// A type alias for the type of collection performed when sweeping references
/// contained in the old generation for the young generation, into the old
/// generation, using the standard process heap configuration
pub type OldSweep<'a> = ReferenceCollection<'a, OldHeap, YoungHeap>;

/// Calculates the reduction count cost of a collection using a rough heuristic
/// for how "expensive" the GC cycle was. This is by no means dialed in - we will
/// likely need to do some testing to find out whether this cost is good enough or
/// too conservative/not conservative enough.
#[inline]
pub fn estimate_cost(moved_live_words: usize, resize_moved_words: usize) -> usize {
    let reds = (moved_live_words / 10) + (resize_moved_words / 100);
    if reds < 1 {
        1
    } else {
        reds
    }
}

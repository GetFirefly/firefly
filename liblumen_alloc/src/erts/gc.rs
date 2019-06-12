mod collector;
mod old_heap;
mod rootset;
mod virtual_heap;
mod young_heap;

/// Represents the types of errors that can occur during garbage collection.
///
/// See the documentation for each variant to get general advice for how to
/// handle these errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcError {
    /// The system is out of memory, and there is not much you can do
    /// but panic, however this choice is left up to the caller
    AllocErr,
    /// Occurs when a process is configured with a maximum heap size,
    /// and a projected heap growth is found to exceed the limit. In
    /// this situation the only meaningful thing to do is to kill the
    /// process
    MaxHeapSizeExceeded,
    /// Indicates that an allocation could not be filled without first
    /// performing a full sweep collection
    FullsweepRequired,
}

pub use self::collector::GarbageCollector;
pub(crate) use self::old_heap::OldHeap;
pub(crate) use self::rootset::RootSet;
pub(crate) use self::virtual_heap::VirtualBinaryHeap;
pub(crate) use self::young_heap::{in_young_gen, YoungHeap};

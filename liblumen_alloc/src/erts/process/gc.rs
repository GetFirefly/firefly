mod collector;
mod old_heap;
mod rootset;
mod virtual_heap;
mod young_heap;

use crate::erts::exception::system::Alloc;

/// Represents the types of errors that can occur during garbage collection.
///
/// See the documentation for each variant to get general advice for how to
/// handle these errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcError {
    /// The system is out of memory, and there is not much you can do
    /// but panic, however this choice is left up to the caller
    Alloc(Alloc),
    /// Occurs when a process is configured with a maximum heap size,
    /// and a projected heap growth is found to exceed the limit. In
    /// this situation the only meaningful thing to do is to kill the
    /// process
    MaxHeapSizeExceeded,
    /// Indicates that an allocation could not be filled without first
    /// performing a full sweep collection
    FullsweepRequired,
}

pub(super) use self::collector::GarbageCollector;
pub(super) use self::old_heap::OldHeap;
pub use self::rootset::RootSet;
pub(super) use self::virtual_heap::VirtualBinaryHeap;
pub(super) use self::young_heap::YoungHeap;

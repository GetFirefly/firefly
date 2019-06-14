use crate::erts::{Term, ProcessControlBlock};

/// This trait represents cloning, like `Clone`, but specifically
/// in the context of terms which need to be cloned into the heap
/// of a specific process, rather than using the global allocator. 
/// 
/// In particular this is used for persistent data structures like
/// `HashMap` which use clone-on-write behavior internally for mutable
/// operations, e.g. `insert`. Rather than using `Clone` which would not
/// do the right thing, we instead implement this trait, and ensure that
/// those operations are provided a mutable reference to the current process
/// so that the clone is into the process heap, rather than the global heap
/// 
/// NOTE: You can implement both `CloneInProcess` and `Clone` for a type,
/// just be aware that any uses of `Clone` will allocate on the global heap
pub trait CloneToProcess {
    /// Returns a copy of this value, performing any heap allocations
    /// using the process heap of `process`, or using heap fragments if
    /// there is not enough space for the cloned value
    fn clone_to_process(&self, process: &mut ProcessControlBlock) -> Term;
}
use super::Heap;

/// An enumeration of the generation types that can be targeted
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Generation {
    Young,
    Old,
}

/// A generational heap is a special form of heap which actually consists
/// of two heaps, one for immature allocations, and another for mature allocations
/// which have survived at least one garbage collection cycle. This is used so
/// that garbage collections are more efficient, as they need only scan the immature
/// heap in the general case, collecting the mature heap only when it is full and
/// a new mature heap is being allocated.
///
/// This trait is intended for use by the garbage collector itself, which is designed
/// for generational heaps, and is responsible for managing the individual heaps as part
/// of its work. For all other use cases, the `Heap` trait is sufficient.
pub trait GenerationalHeap: Heap {
    type Immature: Heap;
    type Mature: Heap;

    /// Returns true if the given pointer is in the immature region of this heap
    fn is_immature(&self, ptr: *const u8) -> bool {
        self.immature().contains(ptr.cast())
    }

    /// Returns true if the given pointer is in the mature region of this heap
    fn is_mature(&self, ptr: *const u8) -> bool {
        self.mature().contains(ptr.cast())
    }

    /// Replaces the current immature heap with a new heap, returning the previous immature heap
    fn swap_immature(&mut self, new_heap: Self::Immature) -> Self::Immature;

    /// Replaces the current mature heap with a new heap, returning the previous mature heap
    fn swap_mature(&mut self, new_heap: Self::Mature) -> Self::Mature;

    /// Get immutable access to the immature heap
    fn immature(&self) -> &Self::Immature;

    /// Get mutable access to the immature heap
    fn immature_mut(&mut self) -> &mut Self::Immature;

    /// Get immutable access to the mature heap
    fn mature(&self) -> &Self::Mature;

    /// Get mutable access to the mature heap
    fn mature_mut(&mut self) -> &mut Self::Mature;
}

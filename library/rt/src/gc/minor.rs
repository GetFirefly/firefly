use firefly_alloc::heap::{Generation, GenerationalHeap, Heap};

use super::*;

/// An implementation of `CollectionType` for minor collections, where
/// references in the young generation of `target` to `source` are swept
/// into either the old generation or young generation of `target`, depending
/// on object maturity. It is expected that the root set has already been swept
/// into the young generation of `target`.
pub struct MinorCollection<'a, S, T>
where
    S: Heap,
    T: GenerationalHeap,
{
    pub source: &'a mut S,
    pub target: &'a mut T,
}
impl<'a, S, T> MinorCollection<'a, S, T>
where
    S: Heap,
    T: GenerationalHeap,
{
    pub fn new(source: &'a mut S, target: &'a mut T) -> Self {
        Self { source, target }
    }

    /// Determine the generation to move the given pointer to
    ///
    /// If `None`, then no move is required
    fn get_generation(&self, ptr: *mut ()) -> Option<Generation> {
        // In a minor collection, we move mature objects into the old generation,
        // otherwise they are moved into the young generation. Objects already in
        // the young/old generation do not need to be moved
        if self.target.contains(ptr) {
            return None;
        }

        // Checking maturity to select destination
        if self.source.mature_range().contains(&ptr.cast()) {
            return Some(Generation::Old);
        }

        // If the object isn't in the mature region, and isn't in the target, then
        // it must be moved into the young generation
        Some(Generation::Young)
    }
}
impl<'a, S, T> CollectionType for MinorCollection<'a, S, T>
where
    S: Heap,
    T: GenerationalHeap,
{
    type Source = S;
    type Target = T;

    fn source(&self) -> &Self::Source {
        self.source
    }

    fn source_mut(&mut self) -> &mut Self::Source {
        self.source
    }

    fn target(&self) -> &Self::Target {
        self.target
    }

    fn target_mut(&mut self) -> &mut Self::Target {
        self.target
    }

    fn should_sweep(&self, ptr: *mut ()) -> bool {
        self.get_generation(ptr).is_some()
    }

    fn sweep_to(&self, ptr: *mut ()) -> &dyn Heap {
        match self.get_generation(ptr).unwrap_or(Generation::Young) {
            Generation::Young => self.target.immature(),
            Generation::Old => self.target.mature(),
        }
    }

    /// This collection will do the following:
    ///
    /// * Trace all live terms from the root set
    /// * If a traced term is not yet in the target heap, it will be swept
    /// into the young generation of the target heap.
    /// * If a traced term is in the young generation of the target heap,
    /// it will be swept into the mature generation of the target heap.
    fn collect(&mut self, mut roots: RootSet) -> Result<usize, GcError> {
        // Follow roots and copy values to appropriate generation
        roots.sweep_roots(self)
    }
}

use firefly_alloc::heap::{GenerationalHeap, Heap};

use log::trace;

use super::*;

/// An implementation of `CollectionType` for full-sweep collections, where
/// references in the target to either generation in the old heap, are swept
/// into the new heap represented by `target`. It is expected that the root
/// set has already been swept into `target`
pub struct FullCollection<'a, S, T>
where
    S: GenerationalHeap,
    T: Heap,
{
    pub source: &'a mut S,
    pub target: &'a mut T,
}
impl<'a, S, T> FullCollection<'a, S, T>
where
    S: GenerationalHeap,
    T: Heap,
{
    pub fn new(source: &'a mut S, target: &'a mut T) -> Self {
        Self { source, target }
    }
}
impl<'a, S, T> CollectionType for FullCollection<'a, S, T>
where
    S: GenerationalHeap,
    T: Heap,
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

    #[inline]
    fn collect(&mut self, mut roots: RootSet) -> Result<usize, GcError> {
        roots.sweep_roots(self)
    }
}

/// Collect all references from `Target` into `Source` by moving the
/// referenced values into `Target`. This is essentially a full collection,
/// but more general as it doesn't assume that the source is a generational
/// heap
pub struct ReferenceCollection<'a, S, T> {
    pub source: &'a mut S,
    pub target: &'a mut T,
}
impl<'a, S, T> ReferenceCollection<'a, S, T>
where
    S: Heap,
    T: Heap,
{
    pub fn new(source: &'a mut S, target: &'a mut T) -> Self {
        Self { source, target }
    }
}
impl<'a, S, T> CollectionType for ReferenceCollection<'a, S, T>
where
    S: Heap,
    T: Heap,
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

    fn collect(&mut self, _roots: RootSet) -> Result<usize, GcError> {
        trace!(target: "process", "sweeping any references from target heap to source heap");
        let mut moved = 0;
        let mut iter = HeapIter::new(self.target.used_range());
        while let Some(ptr) = iter.next() {
            let opaque = unsafe { &mut *ptr };
            if let Move::Ok { bytes_moved, .. } = opaque.sweep(self)? {
                moved += bytes_moved;
            }
        }
        trace!(target: "process", "reference collection complete, moved {} bytes", moved);
        Ok(moved)
    }
}

use crate::erts::process::gc::{CollectionType, GcError, RootSet, Sweep};
use crate::erts::process::term::prelude::Term;

use super::GarbageCollector;

/// This is a bare bones collector which simple executes the
/// given sweep it is created with.
///
/// This is intended only for testing purposes to help isolate
/// issues in lower tiers of the GC infrastructure
pub struct SimpleCollector<T: CollectionType> {
    roots: RootSet,
    gc: T,
    moved: usize,
}
impl<T> SimpleCollector<T>
where
    T: CollectionType,
{
    pub fn new(roots: RootSet, gc: T) -> Self {
        Self {
            roots,
            gc,
            moved: 0,
        }
    }
}
impl<T> GarbageCollector<T> for SimpleCollector<T>
where
    T: Sweep<*mut Term>,
{
    fn garbage_collect(&mut self) -> Result<usize, GcError> {
        use crate::erts::process::gc::collection_type::sweep_root;

        // Follow roots and move values as appropriate
        for mut root in self.roots.iter().copied() {
            let moved = unsafe { sweep_root(&mut self.gc, root.as_mut()) };
            self.moved += moved;
        }

        // Execute underlying gc type now that roots are established
        self.moved += self.gc.collect();

        Ok(self.moved)
    }
}

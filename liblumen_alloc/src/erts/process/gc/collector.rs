mod process_collector;
mod simple_collector;

pub use self::process_collector::ProcessCollector;
pub use self::simple_collector::SimpleCollector;

use super::{CollectionType, GcError};

/// This trait represents the interface for garbage collectors
///
/// A collector returns either `Ok(words_moved)` or `Err(GcError)`
/// upon completion of a collection.
pub trait GarbageCollector<T: CollectionType> {
    /// Execute the collector
    fn garbage_collect(&mut self) -> Result<usize, GcError>;
}

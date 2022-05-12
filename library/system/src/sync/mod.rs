// FairMutex is useful for the kinds of things we'd use spinlocks for.
//
// It guarantees that the lock is fair, which prevents any one thread from starving the others,
// at the cost of some additional overhead.
pub use parking_lot::{const_fair_mutex, FairMutex, FairMutexGuard, MappedFairMutexGuard};

// Mutex/ReentrantMutex are used in place of the libstd Mutexes
pub use parking_lot::{const_mutex, MappedMutexGuard, Mutex, MutexGuard};
pub use parking_lot::{
    const_reentrant_mutex, MappedReentrantMutexGuard, ReentrantMutex, ReentrantMutexGuard,
};

// RwLock is used for read-heavy scenarios where locking is required
pub use parking_lot::{
    const_rwlock, MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard,
    RwLockUpgradableReadGuard, RwLockWriteGuard,
};

// Once is used for one-time initialization use cases
pub use parking_lot::{Once, OnceState};

// This is a type alias for FairMutex to ease refactoring
// It should be removed at some point in the near future.
pub type SpinLock<T> = FairMutex<T>;
pub type SpinLockGuard<'a, T> = FairMutexGuard<'a, T>;

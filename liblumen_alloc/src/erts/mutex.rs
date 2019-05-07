use core::sync::atomic::{AtomicBool, Ordering};
use core::sync::atomic::spin_loop_hint;

use lock_api::{RawMutex, GuardSend};

/// A re-entrant mutex.
/// See docs for `parking_lot::ReentrantMutex<T>` for details
pub type Mutex<T> = parking_lot::ReentrantMutex<T>;

/// A condition variable
/// See docs for `parking_lot::Condvar` for details
pub type Condvar = parking_lot::Condvar;

/// A read/write lock
/// See docs for `parking_lot::RwLock<T>` for details
pub type RwLock<T> = parking_lot::RwLock<T>;

/// Used for simple spinlocks, for locking values, use one of the other mutex types
/// See `lock_api::Mutex<T, U>` for details.
pub type SpinLock<T> = lock_api::Mutex<RawSpinLock, T>;
pub type SpinLockGuard<'a, T> = lock_api::MutexGuard<'a, RawSpinLock, T>;

/// The raw spinlock implementation underlying `SpinLock<T>`
pub struct RawSpinLock(AtomicBool);
impl RawSpinLock {
    pub const fn new(b: bool) -> Self {
        RawSpinLock(AtomicBool::new(b))
    }

    #[cold]
    #[inline(never)]
    fn lock_slow(&self) {
        let mut counter: u32 = 0;
        // Spin until the lock is acquired, but hint to the CPU that we're spinning.
        // The spin uses exponential backoff to help with contention
        loop {
            if self.try_lock() {
                return;
            }
            counter += 1;
            // Cap backoff at 10 iterations, diminishing returns going any higher
            // without adding thread parking to the equation
            if counter > 10 {
                counter = 10;
            }
            for _ in 0..(1 << counter) {
                spin_loop_hint();
            }
        }
    }
}

unsafe impl RawMutex for RawSpinLock {
    const INIT: RawSpinLock = RawSpinLock::new(false);

    type GuardMarker = GuardSend;

    #[inline]
    fn lock(&self) {
        if self.try_lock() {
            return;
        }
        self.lock_slow();
    }

    // Returns true if lock was acquired, false if not
    #[inline]
    fn try_lock(&self) -> bool {
        // If lock is false, then false is returned
        // If lock is true, then true is returned
        // So only return true when we get false (i.e. we acquired the lock)
        self.0.swap(true, Ordering::Acquire) == false
    }

    #[inline]
    fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

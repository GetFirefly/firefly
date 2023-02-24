///! This module exposes the equivalent of `std::sync::Once` and `std::sync::OnceLock`, but usable in `#[no_std]` contexts
use core::cell::UnsafeCell;
use core::fmt;
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::panic::{RefUnwindSafe, UnwindSafe};
use core::pin::Pin;
use core::sync::atomic::{fence, AtomicU8, Ordering};
use parking_lot_core::{self, SpinWait, DEFAULT_PARK_TOKEN, DEFAULT_UNPARK_TOKEN};

/// A synchronization primitive which can be written to only once.
///
/// This type is a thread-safe `OnceCell`.
///
/// # Examples
///
/// ```
/// use firefly_system::sync::OnceLock;
///
/// static CELL: OnceLock<String> = OnceLock::new();
/// assert!(CELL.get().is_none());
///
/// std::thread::spawn(|| {
///     let value: &String = CELL.get_or_init(|| {
///         "Hello, World!".to_string()
///     });
///     assert_eq!(value, "Hello, World!");
/// }).join().unwrap();
///
/// let value: Option<&String> = CELL.get();
/// assert!(value.is_some());
/// assert_eq!(value.unwrap().as_str(), "Hello, World!");
/// ```
pub struct OnceLock<T> {
    once: Once,
    // Whether or not the value is initialized is tracked by `state_and_queue`.
    value: UnsafeCell<MaybeUninit<T>>,
    /// `PhantomData` to make sure dropck understands we're dropping T in our Drop impl.
    ///
    /// ```compile_fail,E0597
    /// use firefly_system::sync::OnceLock;
    ///
    /// struct A<'a>(&'a str);
    ///
    /// impl<'a> Drop for A<'a> {
    ///     fn drop(&mut self) {}
    /// }
    ///
    /// let cell = OnceLock::new();
    /// {
    ///     let s = String::new();
    ///     let _ = cell.set(A(&s));
    /// }
    /// ```
    _marker: PhantomData<T>,
}

impl<T> OnceLock<T> {
    /// Creates a new empty cell.
    #[must_use]
    pub const fn new() -> OnceLock<T> {
        OnceLock {
            once: Once::new(),
            value: UnsafeCell::new(MaybeUninit::uninit()),
            _marker: PhantomData,
        }
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty, or being initialized. This
    /// method never blocks.
    pub fn get(&self) -> Option<&T> {
        if self.is_initialized() {
            // Safe b/c checked is_initialized
            Some(unsafe { self.get_unchecked() })
        } else {
            None
        }
    }

    /// Gets the mutable reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty. This method never blocks.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.is_initialized() {
            // Safe b/c checked is_initialized and we have a unique access
            Some(unsafe { self.get_unchecked_mut() })
        } else {
            None
        }
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// May block if another thread is currently attempting to initialize the cell. The cell is
    /// guaranteed to contain a value when set returns, though not necessarily the one provided.
    ///
    /// Returns `Ok(())` if the cell's value was set by this call.
    ///
    /// # Examples
    ///
    /// ```
    /// use firefly_system::sync::OnceLock;
    ///
    /// static CELL: OnceLock<i32> = OnceLock::new();
    ///
    /// fn main() {
    ///     assert!(CELL.get().is_none());
    ///
    ///     std::thread::spawn(|| {
    ///         assert_eq!(CELL.set(92), Ok(()));
    ///     }).join().unwrap();
    ///
    ///     assert_eq!(CELL.set(62), Err(62));
    ///     assert_eq!(CELL.get(), Some(&92));
    /// }
    /// ```
    pub fn set(&self, value: T) -> Result<(), T> {
        let mut value = Some(value);
        self.get_or_init(|| value.take().unwrap());
        match value {
            None => Ok(()),
            Some(value) => Err(value),
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell
    /// was empty.
    ///
    /// Many threads may call `get_or_init` concurrently with different
    /// initializing functions, but it is guaranteed that only one function
    /// will be executed.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`. The
    /// exact outcome is unspecified. Current implementation deadlocks, but
    /// this may be changed to a panic in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// use firefly_system::sync::OnceLock;
    ///
    /// let cell = OnceLock::new();
    /// let value = cell.get_or_init(|| 92);
    /// assert_eq!(value, &92);
    /// let value = cell.get_or_init(|| unreachable!());
    /// assert_eq!(value, &92);
    /// ```
    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        match self.get_or_try_init(|| Ok::<T, !>(f())) {
            Ok(val) => val,
            Err(_) => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if
    /// the cell was empty. If the cell was empty and `f` failed, an
    /// error is returned.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and
    /// the cell remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`.
    /// The exact outcome is unspecified. Current implementation
    /// deadlocks, but this may be changed to a panic in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// use firefly_system::sync::OnceLock;
    ///
    /// let cell = OnceLock::new();
    /// assert_eq!(cell.get_or_try_init(|| Err(())), Err(()));
    /// assert!(cell.get().is_none());
    /// let value = cell.get_or_try_init(|| -> Result<i32, ()> {
    ///     Ok(92)
    /// });
    /// assert_eq!(value, Ok(&92));
    /// assert_eq!(cell.get(), Some(&92))
    /// ```
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        // Fast path check
        // NOTE: We need to perform an acquire on the state in this method
        // in order to correctly synchronize `LazyLock::force`. This is
        // currently done by calling `self.get()`, which in turn calls
        // `self.is_initialized()`, which in turn performs the acquire.
        if let Some(value) = self.get() {
            return Ok(value);
        }
        self.initialize(f)?;

        debug_assert!(self.is_initialized());

        // SAFETY: The inner value has been initialized
        Ok(unsafe { self.get_unchecked() })
    }

    /// Internal-only API that gets the contents of the cell, initializing it
    /// in two steps with `f` and `g` if the cell was empty.
    ///
    /// `f` is called to construct the value, which is then moved into the cell
    /// and given as a (pinned) mutable reference to `g` to finish
    /// initialization.
    ///
    /// This allows `g` to inspect an manipulate the value after it has been
    /// moved into its final place in the cell, but before the cell is
    /// considered initialized.
    ///
    /// # Panics
    ///
    /// If `f` or `g` panics, the panic is propagated to the caller, and the
    /// cell remains uninitialized.
    ///
    /// With the current implementation, if `g` panics, the value from `f` will
    /// not be dropped. This should probably be fixed if this is ever used for
    /// a type where this matters.
    ///
    /// It is an error to reentrantly initialize the cell from `f`. The exact
    /// outcome is unspecified. Current implementation deadlocks, but this may
    /// be changed to a panic in the future.
    #[allow(unused)]
    pub(crate) fn get_or_init_pin<F, G>(self: Pin<&Self>, f: F, g: G) -> Pin<&T>
    where
        F: FnOnce() -> T,
        G: FnOnce(Pin<&mut T>),
    {
        if let Some(value) = self.get_ref().get() {
            // SAFETY: The inner value was already initialized, and will not be
            // moved anymore.
            return unsafe { Pin::new_unchecked(value) };
        }

        let slot = &self.value;

        // Ignore poisoning from other threads
        // If another thread panics, then we'll be able to run our closure
        self.once.call_once_force(|_| {
            let value = f();
            // SAFETY: We use the Once (self.once) to guarantee unique access
            // to the UnsafeCell (slot).
            let value: &mut T = unsafe { (&mut *slot.get()).write(value) };
            // SAFETY: The value has been written to its final place in
            // self.value. We do not to move it anymore, which we promise here
            // with a Pin<&mut T>.
            g(unsafe { Pin::new_unchecked(value) });
        });

        // SAFETY: The inner value has been initialized, and will not be moved
        // anymore.
        unsafe { Pin::new_unchecked(self.get_ref().get_unchecked()) }
    }

    /// Consumes the `OnceLock`, returning the wrapped value. Returns
    /// `None` if the cell was empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use firefly_system::sync::OnceLock;
    ///
    /// let cell: OnceLock<String> = OnceLock::new();
    /// assert_eq!(cell.into_inner(), None);
    ///
    /// let cell = OnceLock::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.into_inner(), Some("hello".to_string()));
    /// ```
    pub fn into_inner(mut self) -> Option<T> {
        self.take()
    }

    /// Takes the value out of this `OnceLock`, moving it back to an uninitialized state.
    ///
    /// Has no effect and returns `None` if the `OnceLock` hasn't been initialized.
    ///
    /// Safety is guaranteed by requiring a mutable reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use firefly_system::sync::OnceLock;
    ///
    /// let mut cell: OnceLock<String> = OnceLock::new();
    /// assert_eq!(cell.take(), None);
    ///
    /// let mut cell = OnceLock::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.take(), Some("hello".to_string()));
    /// assert_eq!(cell.get(), None);
    /// ```
    pub fn take(&mut self) -> Option<T> {
        if self.is_initialized() {
            self.once = Once::new();
            // SAFETY: `self.value` is initialized and contains a valid `T`.
            // `self.once` is reset, so `is_initialized()` will be false again
            // which prevents the value from being read twice.
            unsafe { Some((&mut *self.value.get()).assume_init_read()) }
        } else {
            None
        }
    }

    #[inline]
    fn is_initialized(&self) -> bool {
        self.once.state().done()
    }

    #[cold]
    fn initialize<F, E>(&self, f: F) -> Result<(), E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let mut res: Result<(), E> = Ok(());
        let slot = &self.value;

        // Ignore poisoning from other threads
        // If another thread panics, then we'll be able to run our closure
        self.once.call_once_force_maybe_poison(|_| {
            match f() {
                Ok(value) => {
                    unsafe { (&mut *slot.get()).write(value) };
                    false
                }
                Err(e) => {
                    res = Err(e);

                    // Treat the underlying `Once` as poisoned since we
                    // failed to initialize our value.
                    true
                }
            }
        });
        res
    }

    /// # Safety
    ///
    /// The value must be initialized
    unsafe fn get_unchecked(&self) -> &T {
        debug_assert!(self.is_initialized());
        (&*self.value.get()).assume_init_ref()
    }

    /// # Safety
    ///
    /// The value must be initialized
    unsafe fn get_unchecked_mut(&mut self) -> &mut T {
        debug_assert!(self.is_initialized());
        (&mut *self.value.get()).assume_init_mut()
    }
}

// Why do we need `T: Send`?
// Thread A creates a `OnceLock` and shares it with
// scoped thread B, which fills the cell, which is
// then destroyed by A. That is, destructor observes
// a sent value.
unsafe impl<T: Sync + Send> Sync for OnceLock<T> {}
unsafe impl<T: Send> Send for OnceLock<T> {}

impl<T: RefUnwindSafe + UnwindSafe> RefUnwindSafe for OnceLock<T> {}
impl<T: UnwindSafe> UnwindSafe for OnceLock<T> {}

impl<T> const Default for OnceLock<T> {
    /// Creates a new empty cell.
    ///
    /// # Example
    ///
    /// ```
    /// use firefly_system::sync::OnceLock;
    ///
    /// fn main() {
    ///     assert_eq!(OnceLock::<()>::new(), OnceLock::default());
    /// }
    /// ```
    fn default() -> OnceLock<T> {
        OnceLock::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for OnceLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => f.debug_tuple("Once").field(v).finish(),
            None => f.write_str("Once(Uninit)"),
        }
    }
}

impl<T: Clone> Clone for OnceLock<T> {
    fn clone(&self) -> OnceLock<T> {
        let cell = Self::new();
        if let Some(value) = self.get() {
            match cell.set(value.clone()) {
                Ok(()) => (),
                Err(_) => unreachable!(),
            }
        }
        cell
    }
}

impl<T> From<T> for OnceLock<T> {
    /// Create a new cell with its contents set to `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use firefly_system::sync::OnceLock;
    ///
    /// # fn main() -> Result<(), i32> {
    /// let a = OnceLock::from(3);
    /// let b = OnceLock::new();
    /// b.set(3)?;
    /// assert_eq!(a, b);
    /// Ok(())
    /// # }
    /// ```
    fn from(value: T) -> Self {
        let cell = Self::new();
        match cell.set(value) {
            Ok(()) => cell,
            Err(_) => unreachable!(),
        }
    }
}

impl<T: PartialEq> PartialEq for OnceLock<T> {
    fn eq(&self, other: &OnceLock<T>) -> bool {
        self.get() == other.get()
    }
}

impl<T: Eq> Eq for OnceLock<T> {}

unsafe impl<#[may_dangle] T> Drop for OnceLock<T> {
    fn drop(&mut self) {
        if self.is_initialized() {
            // SAFETY: The cell is initialized and being dropped, so it can't
            // be accessed again. We also don't touch the `T` other than
            // dropping it, which validates our usage of #[may_dangle].
            unsafe { (&mut *self.value.get()).assume_init_drop() };
        }
    }
}

const DONE_BIT: u8 = 1;
const POISON_BIT: u8 = 2;
const LOCKED_BIT: u8 = 4;
const PARKED_BIT: u8 = 8;

/// Current state of a `Once`.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum OnceState {
    /// A closure has not been executed yet
    New,

    /// A closure was executed but panicked.
    Poisoned,

    /// A thread is currently executing a closure.
    InProgress,

    /// A closure has completed successfully.
    Done,
}

impl OnceState {
    /// Returns whether the associated `Once` has been poisoned.
    ///
    /// Once an initialization routine for a `Once` has panicked it will forever
    /// indicate to future forced initialization routines that it is poisoned.
    #[inline]
    pub fn poisoned(self) -> bool {
        match self {
            OnceState::Poisoned => true,
            _ => false,
        }
    }

    /// Returns whether the associated `Once` has successfully executed a
    /// closure.
    #[inline]
    pub fn done(self) -> bool {
        match self {
            OnceState::Done => true,
            _ => false,
        }
    }
}

/// NOTE: This was ripped from [parking_lot](https://github.com/Amanieu/parking_lot) in
/// order to modify some of the internals to support our `OnceLock` implementation. If at
/// some point in the future we no longer need our `OnceLock`, or the mainline `Once` gains
/// the APIs we need (i.e. `Once::call_once_force_maybe_poison`), then this implementation
/// can be removed.
///
/// A synchronization primitive which can be used to run a one-time
/// initialization. Useful for one-time initialization for globals, FFI or
/// related functionality.
///
/// # Differences from the standard library `Once`
///
/// - Only requires 1 byte of space, instead of 1 word.
/// - Not required to be `'static`.
/// - Relaxed memory barriers in the fast path, which can significantly improve
///   performance on some architectures.
/// - Efficient handling of micro-contention using adaptive spinning.
///
/// # Examples
///
/// ```
/// use firefly_system::sync::Once;
///
/// static START: Once = Once::new();
///
/// START.call_once(|| {
///     // run initialization here
/// });
/// ```
pub struct Once(AtomicU8);

impl Once {
    /// Creates a new `Once` value.
    #[inline]
    pub const fn new() -> Once {
        Once(AtomicU8::new(0))
    }

    /// Returns the current state of this `Once`.
    #[inline]
    pub fn state(&self) -> OnceState {
        let state = self.0.load(Ordering::Acquire);
        if state & DONE_BIT != 0 {
            OnceState::Done
        } else if state & LOCKED_BIT != 0 {
            OnceState::InProgress
        } else if state & POISON_BIT != 0 {
            OnceState::Poisoned
        } else {
            OnceState::New
        }
    }

    /// Performs an initialization routine once and only once. The given closure
    /// will be executed if this is the first time `call_once` has been called,
    /// and otherwise the routine will *not* be invoked.
    ///
    /// This method will block the calling thread if another initialization
    /// routine is currently running.
    ///
    /// When this function returns, it is guaranteed that some initialization
    /// has run and completed (it may not be the closure specified). It is also
    /// guaranteed that any memory writes performed by the executed closure can
    /// be reliably observed by other threads at this point (there is a
    /// happens-before relation between the closure and code executing after the
    /// return).
    ///
    /// # Examples
    ///
    /// ```
    /// use firefly_system::sync::Once;
    ///
    /// static mut VAL: usize = 0;
    /// static INIT: Once = Once::new();
    ///
    /// // Accessing a `static mut` is unsafe much of the time, but if we do so
    /// // in a synchronized fashion (e.g. write once or read all) then we're
    /// // good to go!
    /// //
    /// // This function will only call `expensive_computation` once, and will
    /// // otherwise always return the value returned from the first invocation.
    /// fn get_cached_val() -> usize {
    ///     unsafe {
    ///         INIT.call_once(|| {
    ///             VAL = expensive_computation();
    ///         });
    ///         VAL
    ///     }
    /// }
    ///
    /// fn expensive_computation() -> usize {
    ///     // ...
    /// # 2
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// The closure `f` will only be executed once if this is called
    /// concurrently amongst many threads. If that closure panics, however, then
    /// it will *poison* this `Once` instance, causing all future invocations of
    /// `call_once` to also panic.
    #[inline]
    pub fn call_once<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        if self.0.load(Ordering::Acquire) == DONE_BIT {
            return;
        }

        let mut f = Some(f);
        self.call_once_slow(false, &mut |_| unsafe {
            f.take().unwrap_unchecked()();
            false
        });
    }

    /// Performs the same function as `call_once` except ignores poisoning.
    ///
    /// If this `Once` has been poisoned (some initialization panicked) then
    /// this function will continue to attempt to call initialization functions
    /// until one of them doesn't panic.
    ///
    /// The closure `f` is yielded a structure which can be used to query the
    /// state of this `Once` (whether initialization has previously panicked or
    /// not).
    #[inline]
    pub fn call_once_force<F>(&self, f: F)
    where
        F: FnOnce(OnceState),
    {
        if self.0.load(Ordering::Acquire) == DONE_BIT {
            return;
        }

        let mut f = Some(f);
        self.call_once_slow(true, &mut |state| unsafe {
            f.take().unwrap_unchecked()(state);
            false
        });
    }

    /// Like `call_once_force`, but the given closure can return a boolean that
    /// indicates whether or not to poison the Once even though the closure didn't
    /// poison. This is intended only for internal use in the `OnceLock` implementation
    #[inline]
    fn call_once_force_maybe_poison<F>(&self, f: F)
    where
        F: FnOnce(OnceState) -> bool,
    {
        if self.0.load(Ordering::Acquire) == DONE_BIT {
            return;
        }

        let mut f = Some(f);
        self.call_once_slow(true, &mut |state| unsafe {
            f.take().unwrap_unchecked()(state)
        });
    }

    // This is a non-generic function to reduce the monomorphization cost of
    // using `call_once` (this isn't exactly a trivial or small implementation).
    //
    // Additionally, this is tagged with `#[cold]` as it should indeed be cold
    // and it helps let LLVM know that calls to this function should be off the
    // fast path. Essentially, this should help generate more straight line code
    // in LLVM.
    //
    // Finally, this takes an `FnMut` instead of a `FnOnce` because there's
    // currently no way to take an `FnOnce` and call it via virtual dispatch
    // without some allocation overhead.
    #[cold]
    fn call_once_slow(&self, ignore_poison: bool, f: &mut dyn FnMut(OnceState) -> bool) {
        let mut spinwait = SpinWait::new();
        let mut state = self.0.load(Ordering::Relaxed);
        loop {
            // If another thread called the closure, we're done
            if state & DONE_BIT != 0 {
                // An acquire fence is needed here since we didn't load the
                // state with Ordering::Acquire.
                fence(Ordering::Acquire);
                return;
            }

            // If the state has been poisoned and we aren't forcing, then panic
            if state & POISON_BIT != 0 && !ignore_poison {
                // Need the fence here as well for the same reason
                fence(Ordering::Acquire);
                panic!("Once instance has previously been poisoned");
            }

            // Grab the lock if it isn't locked, even if there is a queue on it.
            // We also clear the poison bit since we are going to try running
            // the closure again.
            if state & LOCKED_BIT == 0 {
                match self.0.compare_exchange_weak(
                    state,
                    (state | LOCKED_BIT) & !POISON_BIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => state = x,
                }
                continue;
            }

            // If there is no queue, try spinning a few times
            if state & PARKED_BIT == 0 && spinwait.spin() {
                state = self.0.load(Ordering::Relaxed);
                continue;
            }

            // Set the parked bit
            if state & PARKED_BIT == 0 {
                if let Err(x) = self.0.compare_exchange_weak(
                    state,
                    state | PARKED_BIT,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    state = x;
                    continue;
                }
            }

            // Park our thread until we are woken up by the thread that owns the
            // lock.
            let addr = self as *const _ as usize;
            let validate = || self.0.load(Ordering::Relaxed) == LOCKED_BIT | PARKED_BIT;
            let before_sleep = || {};
            let timed_out = |_, _| unreachable!();
            unsafe {
                parking_lot_core::park(
                    addr,
                    validate,
                    before_sleep,
                    timed_out,
                    DEFAULT_PARK_TOKEN,
                    None,
                );
            }

            // Loop back and check if the done bit was set
            spinwait.reset();
            state = self.0.load(Ordering::Relaxed);
        }

        struct PanicGuard<'a>(&'a Once);
        impl<'a> Drop for PanicGuard<'a> {
            fn drop(&mut self) {
                // Mark the state as poisoned, unlock it and unpark all threads.
                let once = self.0;
                let state = once.0.swap(POISON_BIT, Ordering::Release);
                if state & PARKED_BIT != 0 {
                    let addr = once as *const _ as usize;
                    unsafe {
                        parking_lot_core::unpark_all(addr, DEFAULT_UNPARK_TOKEN);
                    }
                }
            }
        }

        // At this point we have the lock, so run the closure. Make sure we
        // properly clean up if the closure panicks.
        let guard = PanicGuard(self);
        let once_state = if state & POISON_BIT != 0 {
            OnceState::Poisoned
        } else {
            OnceState::New
        };
        let force_poison = f(once_state);
        mem::forget(guard);

        // Now unlock the state, set the poison/done bit as appropriate and unpark all threads
        let state = if force_poison {
            self.0.swap(POISON_BIT, Ordering::Release)
        } else {
            self.0.swap(DONE_BIT, Ordering::Release)
        };
        if state & PARKED_BIT != 0 {
            let addr = self as *const _ as usize;
            unsafe {
                parking_lot_core::unpark_all(addr, DEFAULT_UNPARK_TOKEN);
            }
        }
    }
}

impl Default for Once {
    #[inline]
    fn default() -> Once {
        Once::new()
    }
}

impl fmt::Debug for Once {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Once")
            .field("state", &self.state())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::Once;
    use std::panic;
    use std::sync::mpsc::channel;
    use std::thread;

    #[test]
    fn smoke_once() {
        static O: Once = Once::new();
        let mut a = 0;
        O.call_once(|| a += 1);
        assert_eq!(a, 1);
        O.call_once(|| a += 1);
        assert_eq!(a, 1);
    }

    #[test]
    fn stampede_once() {
        static O: Once = Once::new();
        static mut RUN: bool = false;

        let (tx, rx) = channel();
        for _ in 0..10 {
            let tx = tx.clone();
            thread::spawn(move || {
                for _ in 0..4 {
                    thread::yield_now()
                }
                unsafe {
                    O.call_once(|| {
                        assert!(!RUN);
                        RUN = true;
                    });
                    assert!(RUN);
                }
                tx.send(()).unwrap();
            });
        }

        unsafe {
            O.call_once(|| {
                assert!(!RUN);
                RUN = true;
            });
            assert!(RUN);
        }

        for _ in 0..10 {
            rx.recv().unwrap();
        }
    }

    #[test]
    fn poison_bad() {
        static O: Once = Once::new();

        // poison the once
        let t = panic::catch_unwind(|| {
            O.call_once(|| panic!());
        });
        assert!(t.is_err());

        // poisoning propagates
        let t = panic::catch_unwind(|| {
            O.call_once(|| {});
        });
        assert!(t.is_err());

        // we can subvert poisoning, however
        let mut called = false;
        O.call_once_force(|p| {
            called = true;
            assert!(p.poisoned())
        });
        assert!(called);

        // once any success happens, we stop propagating the poison
        O.call_once(|| {});
    }

    #[test]
    fn wait_for_force_to_finish() {
        static O: Once = Once::new();

        // poison the once
        let t = panic::catch_unwind(|| {
            O.call_once(|| panic!());
        });
        assert!(t.is_err());

        // make sure someone's waiting inside the once via a force
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        let t1 = thread::spawn(move || {
            O.call_once_force(|p| {
                assert!(p.poisoned());
                tx1.send(()).unwrap();
                rx2.recv().unwrap();
            });
        });

        rx1.recv().unwrap();

        // put another waiter on the once
        let t2 = thread::spawn(|| {
            let mut called = false;
            O.call_once(|| {
                called = true;
            });
            assert!(!called);
        });

        tx2.send(()).unwrap();

        assert!(t1.join().is_ok());
        assert!(t2.join().is_ok());
    }

    #[test]
    fn test_once_debug() {
        static O: Once = Once::new();

        assert_eq!(format!("{:?}", O), "Once { state: New }");
    }
}

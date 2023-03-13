mod collector;
mod full;
mod minor;
mod roots;
mod sweep;

use self::collector::HeapIter;
pub(crate) use self::collector::Reap;
pub use self::collector::SimpleCollector;
pub use self::full::{FullCollection, ReferenceCollection};
pub use self::minor::MinorCollection;
pub use self::roots::{Root, RootSet};
pub use self::sweep::{Move, Sweep};

use firefly_alloc::heap::{Heap, SemispaceHeap};
use firefly_binary::{Aligned, Binary, BinaryFlags, Bitstring, ByteIter, Encoding};

use alloc::alloc::{AllocError, Allocator, Global, Layout};
use core::borrow;
use core::convert::{AsMut, AsRef};
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull, Pointee};

use log::trace;

use crate::error::ExceptionFlags;
use crate::function::ErlangResult;
use crate::process::{ProcessHeap, ProcessLock, StatusFlags};
use crate::term::*;

/// Represents the types of errors that can occur during garbage collection.
///
/// See the documentation for each variant to get general advice for how to
/// handle these errors
#[derive(Debug, PartialEq, Eq)]
pub enum GcError {
    /// The system is out of memory, and there is not much you can do
    /// but panic, however this choice is left up to the caller
    AllocError,
    /// Occurs when a process is configured with a maximum heap size,
    /// and a projected heap growth is found to exceed the limit. In
    /// this situation the only meaningful thing to do is to kill the
    /// process
    MaxHeapSizeExceeded,
    /// Indicates that an allocation could not be filled without first
    /// performing a full sweep collection
    FullsweepRequired,
}
impl From<AllocError> for GcError {
    fn from(_: AllocError) -> Self {
        Self::AllocError
    }
}
impl fmt::Display for GcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::AllocError => f.write_str("unable to allocate memory for garbage collection"),
            Self::MaxHeapSizeExceeded => f.write_str("maximum heap size exceeded"),
            Self::FullsweepRequired => f.write_str("a full garbage collection sweep is required"),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for GcError {}

/// A type alias for `ProcessHeap` when used as a young generation heap
pub type YoungHeap = ProcessHeap;
/// A type alias for `ProcessHeap` when used as an old generation heap
pub type OldHeap = ProcessHeap;

/// A type alias for the semi-space heap which is wrapped by `ProcessHeap`
pub type SemispaceProcessHeap = SemispaceHeap<YoungHeap, OldHeap>;

/// A type alias for the type of a full collection which operates on the standard
/// process heap configuration
pub type FullSweep<'a> = FullCollection<'a, SemispaceProcessHeap, YoungHeap>;
/// A type alias for the type of a minor collection which operates on the standard
/// process heap configuration
pub type MinorSweep<'a> = MinorCollection<'a, YoungHeap, SemispaceProcessHeap>;
/// A type alias for the type of collection performed when sweeping references
/// contained in the old generation for the young generation, into the old
/// generation, using the standard process heap configuration
pub type OldSweep<'a> = ReferenceCollection<'a, OldHeap, YoungHeap>;

/// This trait represents the interface for garbage collectors
///
/// A collector returns either `Ok(words_moved)` or `Err(GcError)`
/// upon completion of a collection.
pub trait GarbageCollector {
    /// Execute the collector
    fn garbage_collect(&mut self, roots: RootSet) -> Result<usize, GcError>;
}

/// Represents a collection algorithm that sweeps for references in `Target`
/// into `Source`, and moving them into `Target`
pub trait CollectionType {
    type Source: Heap;
    type Target: Heap;

    /// Obtain immutable reference to underlying source heap
    fn source(&self) -> &Self::Source;

    /// Obtain mutable reference to underlying source heap
    fn source_mut(&mut self) -> &mut Self::Source;

    /// Obtain immutable reference to underlying target heap
    fn target(&self) -> &Self::Target;

    /// Obtain mutable reference to underlying target heap
    fn target_mut(&mut self) -> &mut Self::Target;

    /// Returns true if `sweepable` should be swept
    ///
    /// By default this always returns true unless overridden.
    fn should_sweep(&self, _sweepable: *mut ()) -> bool {
        true
    }

    /// Returns the heap into which a specific sweepable should be swept.
    ///
    /// This does not imply `should_sweep`, callers should still check if
    /// a sweepable should be swept first. This function instead enables
    /// a collection type to provide different heaps depending on where the
    /// provided sweepable is allocated.
    ///
    /// By default this just returns `target`.
    fn sweep_to(&self, _ptr: *mut ()) -> &dyn Heap {
        self.target()
    }

    /// Performs a collection using an instance of this type
    fn collect(&mut self, roots: RootSet) -> Result<usize, GcError>;
}

/// Garbage collection intrinsic for use by natively-implemented functions
#[inline(never)]
pub fn garbage_collect(process: &mut ProcessLock, roots: RootSet) -> Result<(), ()> {
    use crate::services::error_logger;
    use core::sync::atomic::Ordering;
    use smallvec::SmallVec;

    const MAX_HEAP_ERROR_FORMAT: &'static str = "      Process:            ~p~n\
                                                        Context:            maximum heap size reached~n\
                                                        Max Heap Size:      ~p~n\
                                                        Total Heap Size:    ~p~n\
                                                        Kill:               ~p~n\
                                                        Error Logger:       ~p~n\
                                                        Message Queue Len:  ~p~n\
                                                        GC Info:            ~p~n";

    process.set_status_flags(StatusFlags::GC, Ordering::Relaxed);
    match process.garbage_collect(roots) {
        Ok(reductions) => {
            trace!(
                "garbage collection completed, cost was {} reductions",
                reductions
            );
            // If this GC consumed the remaining reductions, yield, otherwise continue
            process.remove_status_flags(StatusFlags::GC, Ordering::Relaxed);
            process.reductions += reductions;
            Ok(())
        }
        Err(GcError::AllocError) => {
            // Unable to allocate from the system allocator, so try to shut down gracefully
            system_limit_exceeded("out of memory");
        }
        Err(GcError::MaxHeapSizeExceeded) => {
            // GC would require growing the process heap beyond the configured limit,
            // so send an untrappable kill signal to the process
            let max_heap_size = process.max_heap_size();
            trace!(
                "garbage collection could not complete, max heap size of {:?} was reached",
                max_heap_size.size
            );
            if max_heap_size.error_logger {
                // Allocate a heap fragment to hold the format string arguments
                let max_heap_size_in_bytes = match max_heap_size.size {
                    None => Term::Int(0),
                    Some(size) => Term::Int(size.get().try_into().unwrap()),
                };
                let process_heap_size_in_bytes =
                    Term::Int(process.heap.heap_size().try_into().unwrap());
                let signals_len = {
                    let signals = process.signals().lock();
                    Term::Int(signals.len().try_into().unwrap())
                };
                let mut pid = process.pid();
                let mut format_args = SmallVec::<[OpaqueTerm; 8]>::default();
                format_args.push(unsafe { Gc::from_raw(&mut pid) }.into());
                format_args.push(max_heap_size_in_bytes.into());
                format_args.push(process_heap_size_in_bytes.into());
                format_args.push(max_heap_size.kill.into());
                format_args.push(true.into());
                format_args.push(signals_len.into());
                format_args.push(OpaqueTerm::NIL);

                error_logger::send_error_term_to_logger(
                    MAX_HEAP_ERROR_FORMAT,
                    format_args,
                    process.group_leader().cloned(),
                )
                .ok();
            }
            if max_heap_size.kill {
                process.exception_info.flags = ExceptionFlags::EXIT;
                process.exception_info.reason = atoms::Killed.into();
                process.exception_info.value = atoms::Killed.into();
                process.exception_info.trace = None;
                process.stack.nocatch();
                process.set_status_flags(
                    StatusFlags::EXITING | StatusFlags::ACTIVE,
                    Ordering::Release,
                );
                process.remove_status_flags(StatusFlags::GC, Ordering::Release);
                Err(())
            } else {
                Ok(())
            }
        }
        Err(GcError::FullsweepRequired) => unreachable!(),
    }
}

#[export_name = "erlang:garbage_collect/0"]
pub extern "C-unwind" fn garbage_collect0(process: &mut ProcessLock) -> ErlangResult {
    use crate::process::ProcessFlags;

    process.flags |= ProcessFlags::NEED_FULLSWEEP;
    assert!(garbage_collect(process, Default::default()).is_ok());
    ErlangResult::Ok(true.into())
}

#[export_name = "erts_internal:garbage_collect/1"]
pub extern "C-unwind" fn garbage_collect1(
    process: &mut ProcessLock,
    mode_term: OpaqueTerm,
) -> ErlangResult {
    use crate::process::ProcessFlags;

    match mode_term.into() {
        Term::Atom(mode) if mode == atoms::Major => {
            process.flags |= ProcessFlags::NEED_FULLSWEEP;
            assert!(garbage_collect(process, Default::default()).is_ok());
            ErlangResult::Ok(true.into())
        }
        Term::Atom(mode) if mode == atoms::Minor => {
            assert!(garbage_collect(process, Default::default()).is_ok());
            ErlangResult::Ok(true.into())
        }
        _ => {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = mode_term;
            process.exception_info.trace = None;
            ErlangResult::Err
        }
    }
}

#[cfg(feature = "std")]
fn system_limit_exceeded(msg: &str) -> ! {
    std::eprintln!("system limit exceeded: {}", msg);
    std::process::abort();
}

#[cfg(not(feature = "std"))]
fn system_limit_exceeded(_msg: &str) -> ! {
    core::intrinsics::abort();
}

/// Calculates the reduction count cost of a collection using a rough heuristic
/// for how "expensive" the GC cycle was. This is by no means dialed in - we will
/// likely need to do some testing to find out whether this cost is good enough or
/// too conservative/not conservative enough.
#[inline]
pub fn estimate_cost(moved_live_bytes: usize, resize_moved_bytes: usize) -> usize {
    let moved_live_words = moved_live_bytes / core::mem::size_of::<usize>();
    let resize_moved_words = resize_moved_bytes / core::mem::size_of::<usize>();
    let reds = (moved_live_words / 10) + (resize_moved_words / 100);
    if reds < 1 {
        1
    } else {
        reds
    }
}

#[repr(transparent)]
pub struct Gc<T: ?Sized> {
    ptr: NonNull<T>,
}
impl<T> Gc<T> {
    pub fn new(value: T) -> Self {
        Self::new_in(value, &Global).unwrap()
    }

    pub fn new_in<A: ?Sized + Allocator>(value: T, alloc: &A) -> Result<Self, AllocError> {
        let mut this = Self::new_uninit_in(alloc)?;
        this.write(value);
        Ok(unsafe { this.assume_init() })
    }

    pub fn new_uninit() -> Gc<MaybeUninit<T>> {
        Self::new_uninit_in(&Global).unwrap()
    }

    pub fn new_uninit_in<A: ?Sized + Allocator>(
        alloc: &A,
    ) -> Result<Gc<MaybeUninit<T>>, AllocError> {
        let ptr: NonNull<MaybeUninit<T>> = alloc.allocate(Layout::new::<T>())?.cast();
        Ok(Gc { ptr: ptr.cast() })
    }
}
impl<T> Gc<T>
where
    T: ?Sized + Pointee<Metadata = usize>,
{
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_capacity_in(cap, &Global).unwrap()
    }

    pub fn with_capacity_in<A: ?Sized + Allocator>(
        capacity: usize,
        alloc: &A,
    ) -> Result<Self, AllocError> {
        let empty = ptr::from_raw_parts::<T>(ptr::null() as *const (), capacity);
        let layout = unsafe { Layout::for_value_raw(empty) };
        let ptr: NonNull<()> = alloc.allocate(layout)?.cast();
        Ok(Self {
            ptr: NonNull::from_raw_parts(ptr, capacity),
        })
    }

    pub fn with_capacity_zeroed_in<A: ?Sized + Allocator>(
        capacity: usize,
        alloc: &A,
    ) -> Result<Self, AllocError> {
        let empty = ptr::from_raw_parts::<T>(ptr::null() as *const (), capacity);
        let layout = unsafe { Layout::for_value_raw(empty) };
        let ptr: NonNull<()> = alloc.allocate_zeroed(layout)?.cast();
        Ok(Self {
            ptr: NonNull::from_raw_parts(ptr, capacity),
        })
    }
}
impl<T: ?Sized> Gc<T> {
    #[inline]
    pub fn as_ptr(this: &Self) -> *mut () {
        this.ptr.as_ptr().cast()
    }

    #[inline]
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        Self {
            ptr: NonNull::new(ptr).unwrap(),
        }
    }

    #[inline(always)]
    pub const fn as_non_null_ptr(&self) -> NonNull<T> {
        self.ptr
    }

    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut (), metadata: <T as Pointee>::Metadata) -> Self {
        assert!(!ptr.is_null());
        Self {
            ptr: NonNull::from_raw_parts(NonNull::new_unchecked(ptr), metadata),
        }
    }

    #[inline]
    pub fn to_raw_parts(&self) -> (*mut (), <T as Pointee>::Metadata) {
        let (ptr, metadata) = self.ptr.to_raw_parts();
        (ptr.as_ptr(), metadata)
    }
}
impl<T> Gc<MaybeUninit<T>> {
    #[inline]
    pub const unsafe fn assume_init(self) -> Gc<T> {
        Gc {
            ptr: self.ptr.cast(),
        }
    }
}
impl<T: ?Sized> Copy for Gc<T> {}
impl<T: ?Sized> Clone for Gc<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}
impl<T: ?Sized> core::ops::Receiver for Gc<T> {}
impl<T: ?Sized> Deref for Gc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}
impl<T: ?Sized> DerefMut for Gc<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}
impl<T: ?Sized> AsRef<T> for Gc<T> {
    fn as_ref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> AsMut<T> for Gc<T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}
impl<T: ?Sized> borrow::Borrow<T> for Gc<T> {
    fn borrow(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> borrow::BorrowMut<T> for Gc<T> {
    fn borrow_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}
impl<T: ?Sized + fmt::Debug> fmt::Debug for Gc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}
impl<T: ?Sized + fmt::Display> fmt::Display for Gc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}
impl<T: ?Sized> fmt::Pointer for Gc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<T: ?Sized + Eq + PartialEq<Self>> Eq for Gc<T> {}
impl<T, U> PartialEq<U> for Gc<T>
where
    T: ?Sized + PartialEq<U>,
    U: ?Sized,
{
    #[inline]
    fn eq(&self, other: &U) -> bool {
        self.deref().eq(other)
    }

    #[inline]
    fn ne(&self, other: &U) -> bool {
        self.deref().ne(other)
    }
}
impl<T, U> PartialOrd<U> for Gc<T>
where
    T: ?Sized + PartialOrd<U>,
    U: ?Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &U) -> Option<core::cmp::Ordering> {
        self.deref().partial_cmp(other)
    }

    #[inline]
    fn lt(&self, other: &U) -> bool {
        self.deref().lt(other)
    }

    #[inline]
    fn le(&self, other: &U) -> bool {
        self.deref().le(other)
    }

    #[inline]
    fn ge(&self, other: &U) -> bool {
        self.deref().ge(other)
    }

    #[inline]
    fn gt(&self, other: &U) -> bool {
        self.deref().gt(other)
    }
}
impl<T: ?Sized + Ord + PartialOrd<Self>> Ord for Gc<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.deref().cmp(other)
    }
}
impl<T: ?Sized + Hash> Hash for Gc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}
impl<Args, F> FnOnce<Args> for Gc<F>
where
    Args: core::marker::Tuple,
    F: FnOnce<Args> + 'static,
{
    type Output = <F as FnOnce<Args>>::Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output {
        let value = unsafe { ptr::read(self.ptr.as_ptr()) };
        <F as FnOnce<Args>>::call_once(value, args)
    }
}

impl<Args, F> FnMut<Args> for Gc<F>
where
    Args: core::marker::Tuple,
    F: FnMut<Args> + 'static,
{
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output {
        <F as FnMut<Args>>::call_mut(self.deref_mut(), args)
    }
}

impl<Args, F> Fn<Args> for Gc<F>
where
    Args: core::marker::Tuple,
    F: Fn<Args> + 'static,
{
    extern "rust-call" fn call(&self, args: Args) -> Self::Output {
        <F as Fn<Args>>::call(self.deref(), args)
    }
}

impl<T> Bitstring for Gc<T>
where
    T: ?Sized + Bitstring,
{
    #[inline]
    fn byte_size(&self) -> usize {
        self.as_ref().byte_size()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.as_ref().bit_size()
    }

    #[inline]
    fn trailing_bits(&self) -> u8 {
        self.as_ref().trailing_bits()
    }

    #[inline]
    fn bytes(&self) -> ByteIter<'_> {
        self.as_ref().bytes()
    }

    #[inline]
    fn is_aligned(&self) -> bool {
        self.as_ref().is_aligned()
    }

    #[inline]
    fn is_binary(&self) -> bool {
        self.as_ref().is_binary()
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_ref().as_bytes_unchecked()
    }
}

impl<T> Binary for Gc<T>
where
    T: ?Sized + Binary,
{
    #[inline]
    fn flags(&self) -> BinaryFlags {
        self.as_ref().flags()
    }

    #[inline]
    fn is_raw(&self) -> bool {
        self.as_ref().is_raw()
    }

    #[inline]
    fn is_latin1(&self) -> bool {
        self.as_ref().is_latin1()
    }

    #[inline]
    fn is_utf8(&self) -> bool {
        self.as_ref().is_utf8()
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        self.as_ref().encoding()
    }
}

impl<T> Aligned for Gc<T> where T: ?Sized + Aligned {}

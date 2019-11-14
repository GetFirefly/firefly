use core::ops::DerefMut;

use crate::erts::term::prelude::{Term, Boxed, Boxable, ProcBin};

/// Marker trait for virtual allocators
pub trait VirtualAlloc: VirtualHeap<ProcBin> {}

/// This trait represents the core functionality required to
/// allocate, test for membership, and free from a virtual
/// heap.
///
/// It is distinct from `VirtualHeap` due to the fact that some
/// higher level abstractions, such as `ProcessHeap`, may hold
/// a virtual heap, but not expose the full API of a virtual heap
/// expecting such access to remain internal or go through a
/// different set of APIs
pub trait VirtualAllocator<T: Boxable<Term>> {
    /// Adds the given value to the virtual heap
    ///
    /// Since virtual allocation doesn't require ownership, no
    /// value is returned, however the value given is provided
    /// as a boxed value and should be guaranteed by the caller
    /// to be in heap memory so that the underlying value is not
    /// accidentally dropped while still allocated on this virtual heap
    fn virtual_alloc(&mut self, value: Boxed<T>);
    /// Unlinks the given value from the underlying virtual heap,
    /// and executes its destructor, if it has one
    fn virtual_free(&mut self, value: Boxed<T>);
    /// Unlinks the given value from the underlying virtual heap,
    /// but does not drop or read the value
    fn virtual_unlink(&mut self, value: Boxed<T>);
    /// Removes the value referenced by the provided pointer from the underlying
    /// virtual heap, and returns the value on the stack.
    ///
    /// The returned value may then be dropped, or placed on a new virtual
    /// heap. Be aware that the value is moved as a result of this operation,
    /// so any existing pointers to the value will be invalidated. For this
    /// reason, implementations should ensure that the old value contains a
    /// move marker or the none value, this will ensure that an incorrect
    /// usage of the old value cannot result in data corruption.
    ///
    /// NOTE: This operation is only intended to work with pointers to values
    /// allocated on the virtual heap via `virtual_alloc`, callers must verify
    /// that the virtual heap contains the given pointer prior to calling this
    /// function
    fn virtual_pop(&mut self, value: Boxed<T>) -> T;
    /// Returns true if the given pointer is a value on the virtual heap
    ///
    /// Accepts pointers of any type for convenience, but only values of
    /// type `Self::Value` are allowed to be allocated on the underlying
    /// heap
    fn virtual_contains<P: ?Sized>(&self, ptr: *const P) -> bool;
    /// Frees all value references on this virtual heap
    unsafe fn virtual_clear(&mut self);
}

impl<A> VirtualAlloc for A
where
    A: VirtualHeap<ProcBin> {}

impl<A, T, V> VirtualAllocator<T> for A
where
    T: Boxable<Term>,
    V: VirtualAllocator<T>,
    A: DerefMut<Target=V>,
{
    #[inline]
    fn virtual_alloc(&mut self, value: Boxed<T>) {
        self.deref_mut().virtual_alloc(value)
    }

    #[inline]
    fn virtual_free(&mut self, value: Boxed<T>) {
        self.deref_mut().virtual_free(value)
    }

    #[inline]
    fn virtual_unlink(&mut self, value: Boxed<T>) {
        self.deref_mut().virtual_unlink(value)
    }

    #[inline]
    fn virtual_pop(&mut self, value: Boxed<T>) -> T {
        self.deref_mut().virtual_pop(value)
    }

    #[inline]
    fn virtual_contains<P: ?Sized>(&self, ptr: *const P) -> bool {
        self.deref().virtual_contains(ptr)
    }

    #[inline]
    unsafe fn virtual_clear(&mut self) {
        self.deref_mut().virtual_clear()
    }
}

/// This trait is implemented by a heap which exposes virtual allocations,
/// i.e. the objects it tracks are not actually owned by this heap, but are
/// considered owned by the heap by callers of its API.
///
/// Specifically, this trait is used by process heaps to track reference-counted
/// objects that live on the global heap, not the process heap. For purposes
/// of tracking memory usage of the process, as well as making decisions about
/// garbage collection, such objects are "allocated" on a virtual heap. The
/// management of reference counts thus depends on working primarily with boxed
/// references, and only dropping values when removed from the virtual heap and released.
pub trait VirtualHeap<T: Boxable<Term>>: VirtualAllocator<T> {
    /// Gets the total size of this virtual heap
    fn virtual_size(&self) -> usize;
    /// Gets the current amount of virtual heap space used (in bytes)
    fn virtual_heap_used(&self) -> usize;
    /// Gets the current amount of "unused" virtual heap space (in bytes)
    ///
    /// This is a bit of a misnomer, since a virtual heap isn't a real heap,
    /// but the value is used in calculations, such as determining if a garbage
    /// collection should be performed
    fn virtual_heap_unused(&self) -> usize;
}

impl<T, V, B> VirtualHeap<B> for T
where
    B: Boxable<Term>,
    V: VirtualHeap<B>,
    T: DerefMut<Target=V>,
{
    #[inline]
    fn virtual_size(&self) -> usize {
        self.deref().virtual_size()
    }

    #[inline]
    fn virtual_heap_used(&self) -> usize {
        self.deref().virtual_heap_used()
    }

    #[inline]
    fn virtual_heap_unused(&self) -> usize {
        self.deref().virtual_heap_unused()
    }
}

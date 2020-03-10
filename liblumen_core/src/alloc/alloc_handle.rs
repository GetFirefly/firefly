use core::borrow::{Borrow, BorrowMut};
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::{NonNull, Unique};

use crate::alloc::{AllocRef, AllocErr, CannotReallocInPlace, Layout, StaticAlloc};

/// This trait is an extension of `Alloc` which provides a way to obtain
/// a reference to that allocator in a way that abstracts out the type of
/// the reference. Objects allocated with a custom allocator must be
/// associated with the allocator that created them, and so require some
/// kind of reference to be carried around.
///
/// This extra reference would be unnecessary overhead in many cases,
/// such as global allocators, or stateless allocators, where the struct is
/// a zero-sized type. Rather than making the reference a pointer, an `AllocHandle`
/// allows the reference to be implemented as either a zero-sized type, a pointer,
/// or some other type acting as a proxy which is able to calculate the correct
/// allocator based on object pointer/layout information. This allows allocators
/// to specify that their handle is a zero-sized type when possible, and avoid the
/// extra space overhead. Only allocators which require a pointer to locate need
/// store that data in objects they allocate.
///
/// Examples of such stateless allocators are `SysAlloc`, the build-in `System`
/// allocator from `libstd`, and even `MultiBlockCarrier`, which is super-aligned
/// and so pointers to the carrier can be calculated based on the object pointer by
/// masking off the low bits.
///
/// On the other side, `StandardAlloc` requires an actual pointer to be stored,
/// because while it _can_ be global, it is also possible to have multiple instances,
/// so the only way to find the correct instance is to store a pointer in objects it
/// allocates.
pub trait AsAllocHandle<'a> {
    type Handle: AllocHandle<'a>;

    fn as_alloc_handle(&'a self) -> Self::Handle;
}
impl<'a, H: AllocHandle<'a>, T: AsAllocHandle<'a, Handle = H>> AsAllocHandle<'a> for &'a T {
    type Handle = H;

    #[inline]
    fn as_alloc_handle(&'a self) -> Self::Handle {
        (*self).as_alloc_handle()
    }
}

/// A trait to represent a handle to an allocator which can provide
/// immutable and mutable references to the underlying allocator.
///
/// See `Global` and `Handle` for the two types of allocator references
/// provided by this crate
pub trait AllocHandle<'a>: Clone + AllocRef + Sync {
    type AllocRef: ?Sized + AllocRef + Sync;

    fn alloc_handle(&self) -> &Self::AllocRef;
    fn alloc_mut(&mut self) -> &mut Self::AllocRef;
}

/// A zero-sized type for global allocators which only have a single instance
#[derive(PartialEq, Eq)]
pub struct Global<A: 'static>(PhantomData<&'static A>);
impl<A: StaticAlloc> Global<A> {
    #[inline]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<A: StaticAlloc> Clone for Global<A> {
    #[inline]
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}
impl<A: StaticAlloc> AllocHandle<'static> for Global<A> {
    type AllocRef = A;

    #[inline]
    fn alloc_handle(&self) -> &Self::AllocRef {
        self.borrow()
    }

    #[inline]
    fn alloc_mut(&mut self) -> &mut Self::AllocRef {
        self.borrow_mut()
    }
}
unsafe impl<A: StaticAlloc> Sync for Global<A> {}
unsafe impl<A: StaticAlloc> Send for Global<A> {}

impl<A: StaticAlloc> Deref for Global<A> {
    type Target = A;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { <A as StaticAlloc>::static_ref() }
    }
}
impl<A: StaticAlloc> DerefMut for Global<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { <A as StaticAlloc>::static_mut() }
    }
}
impl<A: StaticAlloc> Borrow<A> for Global<A> {
    fn borrow(&self) -> &A {
        unsafe { <A as StaticAlloc>::static_ref() }
    }
}
impl<A: StaticAlloc> BorrowMut<A> for Global<A> {
    fn borrow_mut(&mut self) -> &mut A {
        unsafe { <A as StaticAlloc>::static_mut() }
    }
}
unsafe impl<A: StaticAlloc> AllocRef for Global<A> {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
        <A as StaticAlloc>::static_mut().alloc(layout)
    }
    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        <A as StaticAlloc>::static_mut().dealloc(ptr, layout)
    }

    #[inline]
    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(NonNull<u8>, usize), AllocErr> {
        <A as StaticAlloc>::static_mut().realloc(ptr, layout, new_size)
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
        <A as StaticAlloc>::static_mut().alloc_zeroed(layout)
    }

    #[inline]
    unsafe fn grow_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<usize, CannotReallocInPlace> {
        <A as StaticAlloc>::static_mut().grow_in_place(ptr, layout, new_size)
    }

    #[inline]
    unsafe fn shrink_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<usize, CannotReallocInPlace> {
        <A as StaticAlloc>::static_mut().shrink_in_place(ptr, layout, new_size)
    }
}

/// A pointer-sized type for allocators which may have multiple instances, and
/// so need a handle to the allocator in which an object was allocated
pub struct Handle<'a, A>(Unique<A>, PhantomData<&'a A>);
impl<'a, A: AllocRef + Sync> Handle<'a, A> {
    #[inline]
    pub fn new(alloc: &A) -> Self {
        let nn = unsafe { Unique::new_unchecked(alloc as *const _ as *mut _) };
        Self(nn, PhantomData)
    }
}
impl<'a, A: AllocRef + Sync> Clone for Handle<'a, A> {
    fn clone(&self) -> Self {
        let nn = unsafe { Unique::new_unchecked(self.0.as_ptr()) };
        Self(nn, PhantomData)
    }
}
impl<'a, A: AllocRef + Sync> AllocHandle<'a> for Handle<'a, A> {
    type AllocRef = A;

    #[inline]
    fn alloc_handle(&self) -> &Self::AllocRef {
        self.borrow()
    }

    #[inline]
    fn alloc_mut(&mut self) -> &mut Self::AllocRef {
        self.borrow_mut()
    }
}
impl<'a, A: AllocRef + Sync> Deref for Handle<'a, A> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}
impl<'a, A: AllocRef + Sync> DerefMut for Handle<'a, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut() }
    }
}
impl<'a, A: AllocRef + Sync> Borrow<A> for Handle<'a, A> {
    fn borrow(&self) -> &A {
        unsafe { self.0.as_ref() }
    }
}
impl<'a, A: AllocRef + Sync> BorrowMut<A> for Handle<'a, A> {
    fn borrow_mut(&mut self) -> &mut A {
        unsafe { self.0.as_mut() }
    }
}
unsafe impl<'a, A: AllocRef + Sync> Sync for Handle<'a, A> {}
unsafe impl<'a, A: AllocRef + Sync> Send for Handle<'a, A> {}

unsafe impl<'a, A: AllocRef + Sync> AllocRef for Handle<'a, A> {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
        self.0.as_mut().alloc(layout)
    }
    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        self.0.as_mut().dealloc(ptr, layout);
    }

    #[inline]
    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(NonNull<u8>, usize), AllocErr> {
        self.0.as_mut().realloc(ptr, layout, new_size)
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
        self.0.as_mut().alloc_zeroed(layout)
    }

    #[inline]
    unsafe fn grow_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<usize, CannotReallocInPlace> {
        self.0.as_mut().grow_in_place(ptr, layout, new_size)
    }

    #[inline]
    unsafe fn shrink_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<usize, CannotReallocInPlace> {
        self.0.as_mut().shrink_in_place(ptr, layout, new_size)
    }
}

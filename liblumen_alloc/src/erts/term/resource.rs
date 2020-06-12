/// A resource is something from a BIF or NIF that needs to be memory managed, but cannot be
/// converted to a normal Term.
use core::any::{Any, TypeId};
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display};
use core::ptr::{self, NonNull};
use core::sync::atomic::{self, AtomicUsize};

use liblumen_core::alloc::prelude::Layout;
use liblumen_core::sys::alloc as sys_alloc;

use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::{Heap, TermAlloc};
use crate::CloneToProcess;

use super::prelude::*;

/// A reference-counting smart pointer to a resource handle which
/// can be stored as a term
#[repr(C)]
pub struct Resource {
    header: Header<Resource>,
    inner: NonNull<ResourceInner>,
}
impl_static_header!(Resource, Term::HEADER_RESOURCE_REFERENCE);
impl Resource {
    pub fn new(value: Box<dyn Any>) -> AllocResult<Self> {
        let inner = ResourceInner::new(value)?;
        Ok(Self {
            header: Default::default(),
            inner,
        })
    }

    pub fn from_value<A>(heap: &mut A, value: Box<dyn Any>) -> AllocResult<Boxed<Self>>
    where
        A: ?Sized + Heap,
    {
        let resource = Self::new(value)?;
        let layout = Layout::new::<Self>();

        unsafe {
            let ptr = heap.alloc_layout(layout)?.cast::<Self>().as_ptr();
            ptr.write(resource);

            Ok(Boxed::new_unchecked(ptr))
        }
    }

    /// Try to cast the value of the resource reference to a concrete type
    ///
    /// Returns `None` if the cast is not valid
    #[inline]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.value().downcast_ref()
    }

    /// Returns the inner value, cast to a concrete type, if the `Resource`
    /// is the sole strong reference. Works much like `Arc::try_unwrap`.
    ///
    /// If the value is not able to be cast to the desired type, the operation
    /// fails and `Err` is returned containing the original `Resource` that was
    /// passed in. Likewise, if there are other strong references, `Err` will be
    /// returned.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Result<Box<T>, Self> {
        use atomic::Ordering::{Acquire, Relaxed, Release};
        use core::mem;

        if self.downcast_ref::<T>().is_none() {
            return Err(self);
        }
        if self
            .inner()
            .reference_count
            .compare_exchange(1, 0, Release, Relaxed)
            .is_err()
        {
            return Err(self);
        }

        atomic::fence(Acquire);

        unsafe {
            let inner = self.inner.as_ref();
            let ptr = inner as *const _ as *mut u8;
            let layout = Layout::for_value(&inner);
            let value = ptr::read(&inner.resource);
            // Don't run the Drop impl for this
            mem::forget(self);
            // Just free the allocation for the ResourceInner,
            // which is separate from that of the Box value
            sys_alloc::free(ptr, layout);
            // Cast and return the value
            Ok(value.downcast::<T>().unwrap())
        }
    }

    #[inline]
    pub fn is<T: 'static>(&self) -> bool {
        self.inner().resource.is::<T>()
    }

    #[inline]
    pub fn type_id(&self) -> TypeId {
        self.inner().resource.type_id()
    }

    #[inline]
    pub fn value(&self) -> &dyn Any {
        self.inner().resource.as_ref()
    }

    #[inline]
    fn inner(&self) -> &ResourceInner {
        unsafe { self.inner.as_ref() }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        if self
            .inner()
            .reference_count
            .fetch_sub(1, atomic::Ordering::Release)
            == 1
        {
            atomic::fence(atomic::Ordering::Acquire);
            let inner = self.inner.as_mut();
            // Drop the resource data
            ptr::drop_in_place(&mut inner.resource);
            // Free the allocation for the ResourceInner struct
            let layout = Layout::for_value(&inner);
            sys_alloc::free(inner as *const _ as *mut u8, layout);
        }
    }
}

impl core::hash::Hash for Resource {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

impl PartialEq for Resource {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl<T> PartialEq<Boxed<T>> for Resource
where
    T: PartialEq<Resource>,
{
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}

impl Clone for Resource {
    #[inline]
    fn clone(&self) -> Self {
        self.inner()
            .reference_count
            .fetch_add(1, atomic::Ordering::AcqRel);

        Self {
            header: self.header.clone(),
            inner: self.inner,
        }
    }
}

impl CloneToProcess for Resource {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        self.inner()
            .reference_count
            .fetch_add(1, atomic::Ordering::AcqRel);
        unsafe {
            // Allocate space for the header
            let layout = Layout::new::<Self>();
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            // Write the binary header with an empty link
            ptr::write(
                ptr,
                Self {
                    header: self.header.clone(),
                    inner: self.inner,
                },
            );
            // Reify result term
            Ok(ptr.into())
        }
    }

    fn size_in_words(&self) -> usize {
        crate::erts::to_word_size(Layout::for_value(self).size())
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        if self
            .inner()
            .reference_count
            .fetch_sub(1, atomic::Ordering::Release)
            != 1
        {
            return;
        }
        atomic::fence(atomic::Ordering::Acquire);
        // The refcount is now zero, so we are freeing the memory
        unsafe {
            self.drop_slow();
        }
    }
}

impl Debug for Resource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Resource")
            .field("header", &self.header)
            .field(
                "inner",
                &format_args!(
                    "{:p} => (type_id: {:?}) {:?}",
                    self.inner,
                    self.type_id(),
                    self.value()
                ),
            )
            .finish()
    }
}

impl Display for Resource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Given a raw pointer to the Resource, reborrows and clones it into a new reference.
///
/// # Safety
///
/// This function is unsafe due to dereferencing a raw pointer, but it is expected that
/// this is only ever called with a valid `Resource` pointer anyway. The primary risk
/// with obtaining a `Resource` via this function is if you leak it somehow, rather than
/// letting its `Drop` implementation run. Doing so will leave the reference count greater
/// than 1 forever, meaning memory will never get deallocated.
///
/// NOTE: This does not copy the underlying value, it only obtains a new `Resource`, which is
/// itself a reference to a value held by a `ResourceInner`.
impl From<Boxed<Resource>> for Resource {
    #[inline]
    fn from(boxed: Boxed<Resource>) -> Self {
        boxed.as_ref().clone()
    }
}

impl TryFrom<TypedTerm> for Boxed<Resource> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::ResourceReference(resource) => Ok(resource),
            _ => Err(TypeError),
        }
    }
}

/// A wrapper around a boxed resource value that contains a reference count,
/// similar to `ProcBinInner`
pub struct ResourceInner {
    reference_count: AtomicUsize,
    resource: Box<dyn Any>,
}
impl ResourceInner {
    fn new(resource: Box<dyn Any>) -> AllocResult<NonNull<Self>> {
        let layout = Layout::new::<Self>();

        unsafe {
            let ptr = sys_alloc::alloc(layout)
                .map(|block| block.ptr)
                .map_err(|_| alloc!())?
                .cast::<Self>()
                .as_ptr();

            ptr.write(Self {
                reference_count: Default::default(),
                resource,
            });

            Ok(NonNull::new_unchecked(ptr))
        }
    }
}

impl Debug for ResourceInner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ResourceInner")
            .field("reference_count", &self.reference_count)
            .field(
                "resource",
                &format_args!("Any with {:?}", self.resource.type_id()),
            )
            .finish()
    }
}

impl Display for ResourceInner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

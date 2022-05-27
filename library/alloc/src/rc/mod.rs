use alloc::alloc::{Allocator, Global, Layout};
use alloc::borrow::{self, Cow};
use core::any::{Any, TypeId};
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::marker::{PhantomData, Unsize};
use core::mem::{self, MaybeUninit};
use core::ops::Deref;
use core::ptr::{self, DynMetadata, NonNull, Pointee};
use core::sync::atomic::AtomicUsize;

use static_assertions::assert_eq_size;

use crate::WriteCloneIntoRaw;

/// Represents a non-owning reference to data allocated via `Rc<T>`
///
/// Unlike `std::rc::Rc` and `std::sync::Arc`, there is no actual count of weak references
/// in an `Rc<T>`, instead the weak reference corresponds to the use case where
/// `Copy` semantics are desired in a context where an `Rc<T>` is guaranteed
/// to exist for the lifetime of the weak reference. As a result, there is never
/// a situation in which a weak reference is held without a strong reference, thus
/// no weak count needs to be maintained.
///
/// A weak reference can be converted into a new strong reference (i.e. as if the
/// original `Rc<T>` had been cloned), and is usable in all the same ways as
/// the actual `Rc<T>`, except no changes to the reference count are possible.
pub struct Weak<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    ptr: NonNull<u8>,
    _marker: PhantomData<T>,
}

impl<T> Copy for Weak<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
}

impl<T> Clone for Weak<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T> Weak<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    /// Extracts the TypeId of an opaque pointee
    ///
    /// This is highly unsafe, and is only intended for uses where you are absolutely
    /// certain the pointer given was allocated by Rc.
    ///
    /// In our case, we have opaque pointers on process heaps that are always allocated via Rc,
    /// and we need to be able to cast them back to their original types efficiently. To do so, we
    /// use a combination of this function, a jump table of type ids, and subsequent unchecked casts.
    /// This allows for efficient pointer casts while ensuring we don't improperly cast a pointer to
    /// the wrong type.
    #[inline]
    pub unsafe fn type_id(raw: *mut u8) -> TypeId {
        Rc::<T>::type_id(raw)
    }

    /// Attempts to convert a raw pointer obtained from `Rc::into_raw` back to a `Rc<T>`
    ///
    /// # Safety
    ///
    /// Like `from_raw`, this function is unsafe if the pointer was not allocated by `Rc`.
    ///
    /// NOTE: This function is a bit safer than `from_raw` in that it won't panic if the pointee
    /// is not of the correct type, instead it returns `Err`.
    pub unsafe fn try_from_raw(raw: *mut u8) -> Result<Self, ()> {
        debug_assert!(!raw.is_null());
        let meta = &*header(raw);
        if meta.is::<T>() {
            Ok(Self {
                ptr: NonNull::new_unchecked(raw),
                _marker: PhantomData,
            })
        } else {
            Err(())
        }
    }

    /// Converts a raw pointer obtained from `Rc::into_raw` back to a `Rc<T>`
    ///
    /// # Safety
    ///
    /// This function is unsafe, as there is no way to guarantee that the pointer given
    /// was allocated via `Rc<T>`, and if it wasn't, then calling this function on it
    /// is undefined behavior, and almost certainly bad.
    ///
    /// NOTE: This function will panic if the pointer was allocated for a different type,
    /// but it is always safe to call if the pointer was allocated by `Rc`.
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        debug_assert!(!raw.is_null());
        let raw = raw.cast();
        let meta = &*header(raw);
        assert!(meta.is::<T>());
        Self {
            ptr: NonNull::new_unchecked(raw),
            _marker: PhantomData,
        }
    }

    /// This function is absurdly dangerous.
    ///
    /// However, it exists and is exposed to the rest of this crate for one purpose: to
    /// make pointer casts efficient in the OpaqueTerm -> Term conversion, which specifically
    /// ensures that the pointee is of the correct type before calling this function. It should
    /// not be used _anywhere else_ unless the same guarantees are upheld. For one-off casts from
    /// raw opaque pointers, use `from_raw` or `try_from_raw`, which are slightly less efficient,
    /// but ensure that the pointee is the correct type.
    ///
    /// NOTE: Seriously, don't use this.
    pub unsafe fn from_raw_unchecked(raw: *mut u8) -> Self {
        debug_assert!(!raw.is_null());
        Self {
            ptr: NonNull::new_unchecked(raw),
            _marker: PhantomData,
        }
    }

    /// Upgrades a weak reference to a strong reference, incrementing the strong count
    pub fn upgrade(weak: &Self) -> Rc<T> {
        let strong = mem::ManuallyDrop::new(Rc {
            ptr: weak.ptr,
            _marker: PhantomData,
        });
        Rc::clone(&*strong)
    }

    #[inline]
    pub fn into_raw(boxed: Self) -> *mut T {
        boxed.value()
    }

    #[inline]
    fn value(&self) -> *mut T {
        let meta = unsafe { &*header(self.ptr.as_ptr()) };
        let meta = unsafe { meta.get::<T>() };
        ptr::from_raw_parts_mut(self.ptr.as_ptr().cast(), meta)
    }
}

/// Represents an owned handle to atomically reference-counted data
///
/// Due to the way reference counting of objects allocated on Erlang process
/// heaps works, there are two types of pointers to a ref-counted object:
///
/// * `Rc<T>`, owned, which increments the strong count when cloned, and decrements
/// the strong count when dropped, ultimately freeing the data when the strong count is
/// zero - i.e. typical behavior
/// * `RcRef<T>`, borrowed, which must be derived from an `Rc` known to outlive
/// the lifetime of the `RcRef`. In practice what this means is that when decoding an
/// opaque term which is an `Rc<T>`, we reify it as an `RcRef<T>` instead, only converting
/// it to the owned type when called for. See the `RcRef<T>` docs for more.
pub struct Rc<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    ptr: NonNull<u8>,
    _marker: PhantomData<T>,
}

assert_eq_size!(Rc<[u8]>, *const ());

impl<T> Clone for Rc<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn clone(&self) -> Self {
        let header = unsafe { &*header(self.ptr.as_ptr()) };
        header.increment_strong_count();
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for Rc<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn drop(&mut self) {
        use core::sync::atomic::{self, Ordering};

        let strong_count = {
            let header = unsafe { &*header(self.ptr.as_ptr()) };
            header.decrement_strong_count()
        };
        if strong_count != 1 {
            return;
        }
        // This drop impl is based on Arc<T>, which indicates that this fence
        // is needed to prevent reordering of uses of the data and its deletion.
        // Because the decrement of the refcount is 'Release', it synchronizes with
        // this 'Acquire' fence, which means that any uses of the data must happen before
        // decrementing the refcount, which happens before this fence and therefore before
        // deletion of the data.
        //
        // This is largely due to the fact that ref-counted objects can support interior
        // mutability, so we can't make assumptions based on immutability.
        atomic::fence(Ordering::Acquire);

        unsafe { self.drop_slow() }
    }
}

impl<T> Rc<T>
where
    T: Any,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    pub fn new(value: T) -> Self {
        let meta = Metadata::new::<T>(&value);
        let value_layout = Layout::for_value(&value);
        let (layout, value_offset) = Layout::new::<Metadata>().extend(value_layout).unwrap();
        let ptr: NonNull<u8> = Global.allocate(layout).unwrap().cast();
        unsafe {
            let ptr = NonNull::new_unchecked(ptr.as_ptr().add(value_offset));
            let boxed = Self {
                ptr,
                _marker: PhantomData,
            };
            ptr::write(header(boxed.ptr.as_ptr()), meta);
            ptr::write(boxed.ptr.as_ptr().cast(), value);
            boxed
        }
    }

    pub fn new_uninit() -> Rc<MaybeUninit<T>> {
        let value_layout = Layout::new::<T>();
        let (layout, value_offset) = Layout::new::<Metadata>().extend(value_layout).unwrap();
        let ptr: NonNull<u8> = Global.allocate(layout).unwrap().cast();
        unsafe {
            let ptr = NonNull::new_unchecked(ptr.as_ptr().add(value_offset));
            let meta = Metadata::new::<T>(ptr.as_ptr() as *mut T);
            let boxed = Rc {
                ptr,
                _marker: PhantomData::<MaybeUninit<T>>,
            };
            ptr::write(header(boxed.ptr.as_ptr()), meta);
            boxed
        }
    }
}

impl<T> Rc<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    pub fn new_unsize<V: Unsize<T>>(value: V) -> Self {
        let unsized_: &T = &value;
        let meta = Metadata::new::<T>(unsized_);
        let value_layout = Layout::for_value(&value);
        let (layout, value_offset) = Layout::new::<Metadata>().extend(value_layout).unwrap();
        let ptr: NonNull<u8> = Global.allocate(layout).unwrap().cast();
        unsafe {
            let ptr = NonNull::new_unchecked(ptr.as_ptr().add(value_offset));
            let boxed = Self {
                ptr,
                _marker: PhantomData,
            };
            ptr::write(header(boxed.ptr.as_ptr().cast()), meta);
            let ptr: *mut T = boxed.value();
            ptr::write(ptr.cast(), value);
            boxed
        }
    }

    /// Returns a mutable reference into the given `Rc`, if there are
    /// no other `Rc` pointers to the same allocation.
    ///
    /// Returns `None` otherwise, because it is not safe to mutably alias a shared value.
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            unsafe { Some(Rc::get_mut_unchecked(this)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference into the given `Rc`, without checks.
    ///
    /// This should only ever be called when `this` is known to be a unique reference.
    /// See `get_mut` for a safe alternative.
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        &mut (*this.value())
    }

    /// Gets the number of strong pointers to this allocation
    pub fn strong_count(this: &Self) -> usize {
        let header = unsafe { &*header(this.ptr.as_ptr()) };
        header.strong_count()
    }

    /// Increments the count of strong references to this allocation without cloning the
    /// Rc itself.
    pub fn increment_strong_count(this: &Self) {
        let header = unsafe { &*header(this.ptr.as_ptr()) };
        header.increment_strong_count();
    }

    /// Determine whether this is a unique reference to the underlying data
    #[inline]
    fn is_unique(&mut self) -> bool {
        Self::strong_count(self) == 1
    }

    /// Extracts the TypeId of an opaque pointee
    ///
    /// This is highly unsafe, and is only intended for uses where you are absolutely
    /// certain the pointer given was allocated by Rc.
    ///
    /// In our case, we have opaque pointers on process heaps that are always allocated via Rc,
    /// and we need to be able to cast them back to their original types efficiently. To do so, we
    /// use a combination of this function, a jump table of type ids, and subsequent unchecked casts.
    /// This allows for efficient pointer casts while ensuring we don't improperly cast a pointer to
    /// the wrong type.
    pub unsafe fn type_id(raw: *mut u8) -> TypeId {
        debug_assert!(!raw.is_null());
        let header = &*header(raw);
        header.ty
    }

    /// Attempts to convert a raw pointer obtained from `Rc::into_raw` back to a `Rc<T>`
    ///
    /// # Safety
    ///
    /// Like `from_raw`, this function is unsafe if the pointer was not allocated by `Rc`.
    ///
    /// NOTE: This function is a bit safer than `from_raw` in that it won't panic if the pointee
    /// is not of the correct type, instead it returns `Err`.
    pub unsafe fn try_from_raw(raw: *mut u8) -> Result<Self, ()> {
        debug_assert!(!raw.is_null());
        let meta = &*header(raw);
        if meta.is::<T>() {
            Ok(Self {
                ptr: NonNull::new_unchecked(raw),
                _marker: PhantomData,
            })
        } else {
            Err(())
        }
    }

    /// Converts a raw pointer obtained from `Rc::into_raw` back to a `Rc<T>`
    ///
    /// # Safety
    ///
    /// This function is unsafe, as there is no way to guarantee that the pointer given
    /// was allocated via `Rc<T>`, and if it wasn't, then calling this function on it
    /// is undefined behavior, and almost certainly bad.
    ///
    /// NOTE: This function will panic if the pointer was allocated for a different type,
    /// but it is always safe to call if the pointer was allocated by `Rc`.
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        debug_assert!(!raw.is_null());
        let raw = raw.cast();
        let meta = &*header(raw);
        assert!(meta.is::<T>());
        Self {
            ptr: NonNull::new_unchecked(raw),
            _marker: PhantomData,
        }
    }

    /// This function is absurdly dangerous.
    ///
    /// However, it exists and is exposed to the rest of this crate for one purpose: to
    /// make pointer casts efficient in the OpaqueTerm -> Term conversion, which specifically
    /// ensures that the pointee is of the correct type before calling this function. It should
    /// not be used _anywhere else_ unless the same guarantees are upheld. For one-off casts from
    /// raw opaque pointers, use `from_raw` or `try_from_raw`, which are slightly less efficient,
    /// but ensure that the pointee is the correct type.
    ///
    /// NOTE: Seriously, don't use this.
    pub unsafe fn from_raw_unchecked(raw: *mut u8) -> Self {
        debug_assert!(!raw.is_null());
        Self {
            ptr: NonNull::new_unchecked(raw),
            _marker: PhantomData,
        }
    }

    pub fn into_raw(boxed: Self) -> *mut T {
        let boxed = mem::ManuallyDrop::new(boxed);
        (*boxed).value()
    }

    pub fn into_weak(boxed: Self) -> Weak<T> {
        let raw = Self::into_raw(boxed);
        unsafe { Weak::from_raw_unchecked(raw as *mut u8) }
    }

    #[inline]
    fn value(&self) -> *mut T {
        let meta = unsafe { &*header(self.ptr.as_ptr()) };
        let meta = unsafe { meta.get::<T>() };
        ptr::from_raw_parts_mut(self.ptr.as_ptr().cast(), meta)
    }

    unsafe fn drop_slow(&mut self) {
        let value: *mut T = self.value();
        let (layout, value_offset) = Layout::new::<Metadata>()
            .extend(Layout::for_value_raw(value))
            .unwrap();
        if layout.size() > 0 {
            ptr::drop_in_place(value);
        }
        let ptr = NonNull::new_unchecked(self.ptr.as_ptr().sub(value_offset));
        Global.deallocate(ptr, layout);
    }
}

impl<T> Rc<T>
where
    T: Clone,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    /// Makes a mutable reference into the given `Rc`.
    ///
    /// If there are other `Rc` pointers to the same allocation, then this function will
    /// clone the underlying data to a new allocation to ensure unique ownership, i.e. clone-on-write.
    ///
    /// See `get_mut` for a non-cloning version of this, with the tradeoff that it can only be used on
    /// unique references.
    pub fn make_mut(this: &mut Self) -> &mut T {
        if this.is_unique() {
            unsafe { Self::get_mut_unchecked(this) }
        } else {
            let mut rcbox = Self::new_uninit();
            unsafe {
                let data = Rc::get_mut_unchecked(&mut rcbox);
                (**this).write_clone_into_raw(data.as_mut_ptr());
                *this = rcbox.assume_init();
                Self::get_mut_unchecked(this)
            }
        }
    }
}

impl<T: ?Sized + 'static + Pointee<Metadata = usize>> Rc<T> {
    /// Used to cast an unsized type to its sized representation
    pub fn to_sized<U>(self) -> Rc<U>
    where
        U: Unsize<T> + 'static + Pointee<Metadata = ()>,
    {
        let raw = Self::into_raw(self) as *mut U;
        Rc {
            ptr: unsafe { NonNull::new_unchecked(raw as *mut u8) },
            _marker: PhantomData,
        }
    }
}

impl Rc<dyn Any> {
    /// Attempts to safely cast a Rc<dyn Any> to Rc<T>
    pub fn downcast<T>(self) -> Option<Rc<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &dyn Any = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        Some(Rc {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut u8) },
            _marker: PhantomData,
        })
    }
}

impl Rc<dyn Any + Send> {
    /// Attempts to safely cast a Rc<dyn Any> to Rc<T>
    pub fn downcast<T>(self) -> Option<Rc<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &(dyn Any + Send) = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        Some(Rc {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut u8) },
            _marker: PhantomData,
        })
    }
}

impl Rc<dyn Any + Send + Sync> {
    /// Attempts to safely cast a Rc<dyn Any> to Rc<T>
    pub fn downcast<T>(self) -> Option<Rc<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &(dyn Any + Send + Sync) = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        Some(Rc {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut u8) },
            _marker: PhantomData,
        })
    }
}

impl Weak<dyn Any> {
    /// Attempts to safely cast a Weak<dyn Any> to Weak<T>
    pub fn downcast<T>(self) -> Option<Weak<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &dyn Any = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        Some(Weak {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut u8) },
            _marker: PhantomData,
        })
    }
}

impl Weak<dyn Any + Send> {
    /// Attempts to safely cast a Weak<dyn Any> to Weak<T>
    pub fn downcast<T>(self) -> Option<Weak<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &(dyn Any + Send) = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        Some(Weak {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut u8) },
            _marker: PhantomData,
        })
    }
}

impl Weak<dyn Any + Send + Sync> {
    /// Attempts to safely cast a Weak<dyn Any> to Weak<T>
    pub fn downcast<T>(self) -> Option<Weak<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &(dyn Any + Send + Sync) = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        Some(Weak {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut u8) },
            _marker: PhantomData,
        })
    }
}

impl<T> Rc<T>
where
    T: ?Sized + 'static + Pointee<Metadata = usize>,
{
    pub fn with_capacity(cap: usize) -> Self {
        let empty = ptr::from_raw_parts::<T>(ptr::null() as *const (), cap);
        let meta = Metadata::new::<T>(empty);
        let value_layout = unsafe { Layout::for_value_raw(empty) };
        let (layout, value_offset) = Layout::new::<Metadata>().extend(value_layout).unwrap();
        let ptr: NonNull<u8> = Global.allocate(layout).unwrap().cast();
        unsafe {
            let ptr = NonNull::new_unchecked(ptr.as_ptr().add(value_offset));
            let boxed = Self {
                ptr,
                _marker: PhantomData,
            };
            ptr::write(header(boxed.ptr.as_ptr()), meta);
            boxed
        }
    }
}

impl<T> Rc<[T]> {
    #[inline]
    pub fn new_uninit_slice(len: usize) -> Rc<[MaybeUninit<T>]> {
        Rc::<[MaybeUninit<T>]>::with_capacity(len)
    }
}

impl<T: Clone> Rc<[T]> {
    pub fn to_boxed_slice(slice: &[T]) -> Self {
        let mut boxed = Rc::<[MaybeUninit<T>]>::with_capacity(slice.len());
        boxed.write_slice_cloned(slice);
        unsafe { boxed.assume_init() }
    }
}

impl<T: Any> Rc<MaybeUninit<T>> {
    pub unsafe fn assume_init(self) -> Rc<T> {
        let raw = Rc::into_raw(self);
        Rc::from_raw(raw as *mut T)
    }

    pub fn write(boxed: Self, value: T) -> Rc<T> {
        unsafe {
            (&mut *boxed.value()).write(value);
            boxed.assume_init()
        }
    }
}

impl<T> Rc<[MaybeUninit<T>]> {
    pub unsafe fn assume_init(self) -> Rc<[T]> {
        let raw = Rc::into_raw(self);
        Rc::from_raw(raw as *mut [T])
    }
}

impl<T: Copy> Rc<[MaybeUninit<T>]> {
    pub fn write_slice(&mut self, src: &[T]) {
        MaybeUninit::write_slice(unsafe { &mut *self.value() }, src);
    }
}

impl<T: Clone> Rc<[MaybeUninit<T>]> {
    pub fn write_slice_cloned(&mut self, src: &[T]) {
        MaybeUninit::write_slice_cloned(unsafe { &mut *self.value() }, src);
    }
}

impl<T> Debug for Rc<T>
where
    T: ?Sized + 'static + Debug,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T> Debug for Weak<T>
where
    T: ?Sized + 'static + Debug,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T> Display for Rc<T>
where
    T: ?Sized + 'static + Display,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T> Display for Weak<T>
where
    T: ?Sized + 'static + Display,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T> fmt::Pointer for Rc<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<T> fmt::Pointer for Weak<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<T> Deref for Rc<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        let ptr = self.value();
        unsafe { &*ptr }
    }
}

impl<T> Deref for Weak<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        let ptr = self.value();
        unsafe { &*ptr }
    }
}

impl<T> borrow::Borrow<T> for Rc<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T> borrow::Borrow<T> for Weak<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T> AsRef<T> for Rc<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T> AsRef<T> for Weak<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T> PartialEq for Rc<T>
where
    T: ?Sized + PartialEq + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }

    #[inline]
    fn ne(&self, other: &Self) -> bool {
        PartialEq::ne(&**self, &**other)
    }
}

impl<T> PartialEq<T> for Rc<T>
where
    T: ?Sized + PartialEq + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn eq(&self, other: &T) -> bool {
        PartialEq::eq(&**self, other)
    }

    #[inline]
    fn ne(&self, other: &T) -> bool {
        PartialEq::ne(&**self, other)
    }
}

impl<T> Eq for Rc<T>
where
    T: ?Sized + 'static + Eq,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
}

impl<T> PartialEq for Weak<T>
where
    T: ?Sized + PartialEq + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }

    #[inline]
    fn ne(&self, other: &Self) -> bool {
        PartialEq::ne(&**self, &**other)
    }
}

impl<T> PartialEq<T> for Weak<T>
where
    T: ?Sized + PartialEq + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn eq(&self, other: &T) -> bool {
        PartialEq::eq(&**self, other)
    }

    #[inline]
    fn ne(&self, other: &T) -> bool {
        PartialEq::ne(&**self, other)
    }
}

impl<T> Eq for Weak<T>
where
    T: ?Sized + 'static + Eq,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
}

impl<T> PartialOrd<T> for Rc<T>
where
    T: ?Sized + 'static + PartialOrd,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, other)
    }

    #[inline]
    fn lt(&self, other: &T) -> bool {
        PartialOrd::lt(&**self, other)
    }

    #[inline]
    fn le(&self, other: &T) -> bool {
        PartialOrd::le(&**self, other)
    }

    #[inline]
    fn ge(&self, other: &T) -> bool {
        PartialOrd::ge(&**self, other)
    }

    #[inline]
    fn gt(&self, other: &T) -> bool {
        PartialOrd::gt(&**self, other)
    }
}

impl<T> PartialOrd for Rc<T>
where
    T: ?Sized + 'static + PartialOrd,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        PartialOrd::lt(&**self, &**other)
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        PartialOrd::le(&**self, &**other)
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        PartialOrd::ge(&**self, &**other)
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}

impl<T> Ord for Rc<T>
where
    T: ?Sized + 'static + Ord,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T> PartialOrd<T> for Weak<T>
where
    T: ?Sized + 'static + PartialOrd,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, other)
    }

    #[inline]
    fn lt(&self, other: &T) -> bool {
        PartialOrd::lt(&**self, other)
    }

    #[inline]
    fn le(&self, other: &T) -> bool {
        PartialOrd::le(&**self, other)
    }

    #[inline]
    fn ge(&self, other: &T) -> bool {
        PartialOrd::ge(&**self, other)
    }

    #[inline]
    fn gt(&self, other: &T) -> bool {
        PartialOrd::gt(&**self, other)
    }
}

impl<T> PartialOrd for Weak<T>
where
    T: ?Sized + 'static + PartialOrd,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        PartialOrd::lt(&**self, &**other)
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        PartialOrd::le(&**self, &**other)
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        PartialOrd::ge(&**self, &**other)
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}

impl<T> Ord for Weak<T>
where
    T: ?Sized + 'static + Ord,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T> Hash for Rc<T>
where
    T: ?Sized + 'static + Hash,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T> Hash for Weak<T>
where
    T: ?Sized + 'static + Hash,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T> From<T> for Rc<T>
where
    T: Any,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn from(t: T) -> Self {
        Rc::new(t)
    }
}

impl<T: Copy> From<&[T]> for Rc<[T]> {
    fn from(slice: &[T]) -> Self {
        let mut boxed = Rc::<[MaybeUninit<T>]>::with_capacity(slice.len());
        boxed.write_slice(slice);
        unsafe { boxed.assume_init() }
    }
}

impl<T: Copy> From<Cow<'_, [T]>> for Rc<[T]> {
    #[inline]
    fn from(cow: Cow<'_, [T]>) -> Self {
        match cow {
            Cow::Borrowed(slice) => Rc::from(slice),
            Cow::Owned(slice) => Rc::from(slice.as_slice()),
        }
    }
}

fn header(ptr: *mut u8) -> *mut Metadata {
    unsafe { ptr.sub(mem::size_of::<Metadata>()).cast() }
}

/// This metadata provides enough information to restore a fat pointer from a thin
/// pointer, and to cast to and from Opaque
pub struct Metadata {
    refc: AtomicUsize,
    ty: TypeId,
    meta: PtrMetadata,
}
impl Metadata {
    fn new<T: ?Sized + 'static>(raw: *const T) -> Self
    where
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let ty = TypeId::of::<T>();
        let meta = ptr::metadata(raw);
        Self {
            refc: AtomicUsize::new(1),
            ty,
            meta: meta.into(),
        }
    }

    #[inline(always)]
    fn strong_count(&self) -> usize {
        use core::sync::atomic::Ordering;

        self.refc.load(Ordering::SeqCst)
    }

    #[inline]
    fn increment_strong_count(&self) {
        use core::sync::atomic::Ordering;

        // In order to increment the refcount, we must already have a reference,
        // which prevents destruction of the data, so use of 'Relaxed' is always
        // acceptable
        self.refc.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn decrement_strong_count(&self) -> usize {
        use core::sync::atomic::Ordering;

        // This 'Release' ordering synchronizes with the fence in `drop`
        self.refc.fetch_sub(1, Ordering::Release)
    }

    /// This function attempts to extract pointer metadata valid for a pointer to type T
    /// from the current metadata contained in this struct. This is unsafe in that it allows
    /// coercions between any types that use the same metadata. If coercions between those types
    /// is safe, then this is fine; but if the coercions imply a different layout in memory, then
    /// this function is absolutely not safe to be called. It is up to the caller to ensure the
    /// following:
    ///
    /// * This function is only called with the same type as was used to produce the metadata,
    /// or a type for whom the metadata is equivalent and implies the same layout in memory.
    ///
    /// NOTE: Invalid casts between different metadata types are caught, i.e. it is not possible
    /// to read unsized metadata for a sized type or vice versa, same with trait objects. However,
    /// the risk is with reading unsized metadata or trait object metadata with incorrect types.
    unsafe fn get<T>(&self) -> <T as Pointee>::Metadata
    where
        T: ?Sized + Pointee,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        self.meta
            .try_into()
            .unwrap_or_else(|_| panic!("invalid pointer metadata"))
    }

    /// Returns true if this metadata is for a value of type T
    #[inline]
    fn is<T: ?Sized + Any>(&self) -> bool {
        self.ty == TypeId::of::<T>()
    }
}

assert_eq_size!(PtrMetadata, usize);

#[derive(Copy, Clone)]
pub union PtrMetadata {
    size: usize,
    dynamic: DynMetadata<dyn Any>,
    dyn_send: DynMetadata<dyn Any + Send>,
    dyn_send_sync: DynMetadata<dyn Any + Send + Sync>,
}
impl PtrMetadata {
    const UNSIZED_FLAG: usize = 1 << ((core::mem::size_of::<usize>() * 8) - 1);

    #[inline]
    fn is_sized(&self) -> bool {
        unsafe { self.size == 0 }
    }

    #[inline]
    fn is_unsized(&self) -> bool {
        unsafe { self.size & Self::UNSIZED_FLAG == Self::UNSIZED_FLAG }
    }

    #[inline]
    fn is_dyn(&self) -> bool {
        !(self.is_sized() || self.is_unsized())
    }
}
impl From<()> for PtrMetadata {
    fn from(_: ()) -> Self {
        Self { size: 0 }
    }
}
impl From<usize> for PtrMetadata {
    fn from(size: usize) -> Self {
        Self {
            size: (size | Self::UNSIZED_FLAG),
        }
    }
}
impl From<DynMetadata<dyn Any>> for PtrMetadata {
    fn from(dynamic: DynMetadata<dyn Any>) -> Self {
        let meta = Self { dynamic };
        let raw = unsafe { meta.size };
        assert_ne!(
            raw & Self::UNSIZED_FLAG,
            Self::UNSIZED_FLAG,
            "dyn metadata conflicts with sized metadata flag"
        );
        meta
    }
}
impl From<DynMetadata<dyn Any + Send>> for PtrMetadata {
    fn from(dyn_send: DynMetadata<dyn Any + Send>) -> Self {
        let meta = Self { dyn_send };
        let raw = unsafe { meta.size };
        assert_ne!(
            raw & Self::UNSIZED_FLAG,
            Self::UNSIZED_FLAG,
            "dyn metadata conflicts with sized metadata flag"
        );
        meta
    }
}
impl From<DynMetadata<dyn Any + Send + Sync>> for PtrMetadata {
    fn from(dyn_send_sync: DynMetadata<dyn Any + Send + Sync>) -> Self {
        let meta = Self { dyn_send_sync };
        let raw = unsafe { meta.size };
        assert_ne!(
            raw & Self::UNSIZED_FLAG,
            Self::UNSIZED_FLAG,
            "dyn metadata conflicts with sized metadata flag"
        );
        meta
    }
}
impl TryInto<()> for PtrMetadata {
    type Error = core::convert::Infallible;
    #[inline]
    fn try_into(self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl TryInto<usize> for PtrMetadata {
    type Error = InvalidPtrMetadata;
    #[inline]
    fn try_into(self) -> Result<usize, Self::Error> {
        if self.is_unsized() {
            Ok(unsafe { self.size } & !Self::UNSIZED_FLAG)
        } else {
            Err(InvalidPtrMetadata)
        }
    }
}
impl TryInto<DynMetadata<dyn Any>> for PtrMetadata {
    type Error = InvalidPtrMetadata;
    #[inline]
    fn try_into(self) -> Result<DynMetadata<dyn Any>, Self::Error> {
        if self.is_dyn() {
            Ok(unsafe { self.dynamic })
        } else {
            Err(InvalidPtrMetadata)
        }
    }
}
impl TryInto<DynMetadata<dyn Any + Send>> for PtrMetadata {
    type Error = InvalidPtrMetadata;
    #[inline]
    fn try_into(self) -> Result<DynMetadata<dyn Any + Send>, Self::Error> {
        if self.is_dyn() {
            Ok(unsafe { self.dyn_send })
        } else {
            Err(InvalidPtrMetadata)
        }
    }
}
impl TryInto<DynMetadata<dyn Any + Send + Sync>> for PtrMetadata {
    type Error = InvalidPtrMetadata;
    #[inline]
    fn try_into(self) -> Result<DynMetadata<dyn Any + Send + Sync>, Self::Error> {
        if self.is_dyn() {
            Ok(unsafe { self.dyn_send_sync })
        } else {
            Err(InvalidPtrMetadata)
        }
    }
}

pub struct InvalidPtrMetadata;

#[cfg(test)]
mod test {
    use super::*;

    struct Cons {
        head: usize,
        tail: usize,
    }

    struct Tuple {
        _header: usize,
        data: [usize],
    }
    impl Tuple {
        fn len(&self) -> usize {
            self.data.len()
        }
        fn set_element(&mut self, index: usize, value: usize) {
            self.data[index] = value;
        }
        fn get_element(&self, index: usize) -> usize {
            self.data[index]
        }
        fn get(&self, index: usize) -> Option<usize> {
            self.data.get(index).copied()
        }
    }

    #[test]
    fn can_rcbox_pods() {
        let value = Rc::new(Cons { head: 1, tail: 2 });
        assert_eq!(value.head, 1);
        assert_eq!(value.tail, 2);
    }

    #[test]
    fn can_rcbox_dsts() {
        let mut rcbox = Rc::<Tuple>::with_capacity(2);
        let value = Rc::get_mut(&mut rcbox).unwrap();
        value.set_element(0, 42);
        value.set_element(1, 11);
        assert_eq!(value.len(), 2);
        assert_eq!(value.get_element(0), 42);
        assert_eq!(value.get_element(1), 11);
    }

    #[test]
    fn can_rcbox_downcast_any() {
        let value = Rc::<dyn Any>::new_unsize(Cons { head: 1, tail: 2 });
        assert!(value.is::<Cons>());
        let result = value.downcast::<Cons>();
        assert!(result.is_some());
        let cons = result.unwrap();
        assert_eq!(cons.head, 1);
        assert_eq!(cons.tail, 2);
    }

    #[test]
    fn cant_rcbox_downcast_any_improperly() {
        let value = Rc::<dyn Any>::new_unsize(Cons { head: 1, tail: 2 });
        assert!(!value.is::<usize>());
        let result = value.downcast::<usize>();
        assert!(result.is_none());
    }

    #[test]
    fn rcbox_clone_and_drop_correctly_modifies_refcount() {
        let first = Rc::new(Cons { head: 1, tail: 2 });
        {
            let second = first.clone();
            assert_eq!(first.strong_count(), 2);
        }
        assert_eq!(first.strong_count(), 1);
    }

    #[test]
    fn rcbox_make_mut() {
        // A unique reference doesn't clone
        let mut first = Rc::new(Cons { head: 1, tail: 2 });
        *Rc::make_mut(&mut first).head = 2;
        assert_eq!(first.strong_count(), 1);
        assert_eq!(first.head, 2);
        // But a non-unique references causes a clone
        let mut second = first.clone();
        *Rc::make_mut(&mut second).head = 3;
        assert_eq!(first.strong_count(), 1);
        assert_eq!(first.head, 2);
        assert_eq!(second.strong_count(), 1);
        assert_eq!(second.head, 3);
    }
}

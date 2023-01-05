use alloc::alloc::{AllocError, Allocator, Global, Layout};
use alloc::borrow::{self, Cow};
use core::any::{Any, TypeId};
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::marker::{PhantomData, Unsize};
use core::mem::{self, MaybeUninit};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, DynMetadata, NonNull, Pointee};

use static_assertions::assert_eq_size;

#[repr(transparent)]
pub struct GcBox<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    ptr: NonNull<()>,
    _marker: PhantomData<T>,
}

assert_eq_size!(GcBox<dyn Any>, *const ());

impl<T: ?Sized + 'static> Copy for GcBox<T> where
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>
{
}

impl<T: ?Sized + 'static> Clone for GcBox<T>
where
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T> GcBox<T>
where
    T: Any,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    pub fn new(value: T) -> Self {
        Self::new_in(value, Global).unwrap()
    }

    pub fn new_uninit() -> GcBox<MaybeUninit<T>> {
        Self::new_uninit_in(Global).unwrap()
    }

    pub fn new_in<A: Allocator>(value: T, alloc: A) -> Result<Self, AllocError> {
        let meta = Metadata::new::<T>(&value);
        let value_layout = Layout::for_value(&value);
        let (layout, value_offset) = Layout::new::<Metadata>().extend(value_layout).unwrap();
        let ptr: NonNull<()> = alloc.allocate(layout)?.cast();
        unsafe {
            let ptr = NonNull::new_unchecked(ptr.as_ptr().byte_add(value_offset));
            let boxed = Self {
                ptr,
                _marker: PhantomData,
            };
            ptr::write(header(boxed.ptr.as_ptr()), meta);
            ptr::write(boxed.ptr.as_ptr().cast(), value);
            Ok(boxed)
        }
    }

    pub fn new_uninit_in<A: Allocator>(alloc: A) -> Result<GcBox<MaybeUninit<T>>, AllocError> {
        let value_layout = Layout::new::<T>();
        let (layout, value_offset) = Layout::new::<Metadata>().extend(value_layout).unwrap();
        let ptr: NonNull<()> = alloc.allocate(layout)?.cast();
        unsafe {
            let ptr = NonNull::new_unchecked(ptr.as_ptr().byte_add(value_offset));
            let meta = Metadata::new::<T>(ptr.as_ptr() as *mut T);
            let boxed = GcBox {
                ptr,
                _marker: PhantomData::<MaybeUninit<T>>,
            };
            ptr::write(header(boxed.ptr.as_ptr()), meta);
            Ok(boxed)
        }
    }
}

impl<T> GcBox<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    pub fn new_unsize<V: Unsize<T>>(value: V) -> Self {
        Self::new_unsize_in(value, Global).unwrap()
    }

    pub fn new_unsize_in<V: Unsize<T>, A: Allocator>(
        value: V,
        alloc: A,
    ) -> Result<Self, AllocError> {
        let unsized_: &T = &value;
        let meta = Metadata::new::<T>(unsized_);
        let value_layout = Layout::for_value(&value);
        let (layout, value_offset) = Layout::new::<Metadata>().extend(value_layout).unwrap();
        let ptr: NonNull<()> = alloc.allocate(layout)?.cast();
        unsafe {
            let ptr = NonNull::new_unchecked(ptr.as_ptr().byte_add(value_offset));
            let boxed = Self {
                ptr,
                _marker: PhantomData,
            };
            ptr::write(header(boxed.ptr.as_ptr().cast()), meta);
            let ptr: *mut T = boxed.value();
            ptr::write(ptr.cast(), value);
            Ok(boxed)
        }
    }

    /// Converts a GcBox<T> to GcBox<U> where T: Unsize<U>
    ///
    /// This is the only type of in-place conversion that is safe for GcBox,
    /// based on the guarantees provided by the Unsize marker trait.
    pub fn cast<U>(self) -> GcBox<U>
    where
        U: ?Sized + 'static,
        T: Unsize<U>,
        PtrMetadata: From<<U as Pointee>::Metadata> + TryInto<<U as Pointee>::Metadata>,
    {
        // Get the raw pointer to T
        let ptr = Self::into_raw(self);
        // Coerce via Unsize<U> to U, and get the new metadata
        let meta = {
            let unsized_: &U = unsafe { &*ptr };
            Metadata::new::<U>(unsized_)
        };
        // Construct a new box
        let boxed = GcBox {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut ()) },
            _marker: PhantomData,
        };
        // Write the new metadata to the header for our new box
        let header = header(boxed.ptr.as_ptr());
        unsafe {
            *header = meta;
        }
        boxed
    }

    /// Extracts the TypeId of an opaque pointee
    ///
    /// This is highly unsafe, and is only intended for uses where you are absolutely
    /// certain the pointer given was allocated by GcBox.
    ///
    /// In our case, we have opaque pointers on process heaps that are always allocated via GcBox,
    /// and we need to be able to cast them back to their original types efficiently. To do so, we
    /// use a combination of this function, a jump table of type ids, and subsequent unchecked
    /// casts. This allows for efficient pointer casts while ensuring we don't improperly cast a
    /// pointer to the wrong type.
    pub unsafe fn type_id(raw: *mut ()) -> TypeId {
        debug_assert!(!raw.is_null());
        let header = &*header(raw);
        header.ty
    }

    /// Attempts to convert a raw pointer obtained from `GcBox::into_raw` back to a `GcBox<T>`
    ///
    /// # Safety
    ///
    /// Like `from_raw`, this function is unsafe if the pointer was not allocated by `GcBox`.
    ///
    /// NOTE: This function is a bit safer than `from_raw` in that it won't panic if the pointee
    /// is not of the correct type, instead it returns `Err`.
    pub unsafe fn try_from_raw(raw: *mut ()) -> Result<Self, ()> {
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

    /// Converts a raw pointer obtained from `GcBox::into_raw` back to a `GcBox<T>`
    ///
    /// # Safety
    ///
    /// This function is unsafe, as there is no way to guarantee that the pointer given
    /// was allocated via `GcBox<T>`, and if it wasn't, then calling this function on it
    /// is undefined behavior, and almost certainly bad.
    ///
    /// NOTE: This function will panic if the pointer was allocated for a different type,
    /// but it is always safe to call if the pointer was allocated by `GcBox`.
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
    /// make pointer casts efficient in the OpaqueTerm -> Value conversion, which specifically
    /// ensures that the pointee is of the correct type before calling this function. It should
    /// not be used _anywhere else_ unless the same guarantees are upheld. For one-off casts from
    /// raw opaque pointers, use `from_raw` or `try_from_raw`, which are slightly less efficient,
    /// but ensure that the pointee is the correct type.
    ///
    /// NOTE: Seriously, don't use this.
    pub unsafe fn from_raw_unchecked(raw: *mut ()) -> Self {
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

    #[inline]
    pub fn as_ptr(boxed: &Self) -> *mut () {
        boxed.ptr.as_ptr()
    }

    pub unsafe fn drop_in<A: Allocator>(boxed: Self, alloc: A) {
        let value: *mut T = boxed.value();
        let (layout, value_offset) = Layout::new::<Metadata>()
            .extend(Layout::for_value_raw(value))
            .unwrap();
        if layout.size() > 0 {
            ptr::drop_in_place(value);
        }
        let ptr = NonNull::new_unchecked(boxed.ptr.as_ptr().byte_sub(value_offset));
        alloc.deallocate(ptr.cast(), layout);
    }

    #[inline]
    fn value(&self) -> *mut T {
        let meta = unsafe { *header(self.ptr.as_ptr()) };
        let meta = unsafe { meta.get::<T>() };
        ptr::from_raw_parts_mut(self.ptr.as_ptr().cast(), meta)
    }
}

impl<T: ?Sized + 'static + Pointee<Metadata = usize>> GcBox<T> {
    /// Used to cast an unsized type to its sized representation
    pub fn to_sized<U>(self) -> GcBox<U>
    where
        U: Unsize<T> + 'static + Pointee<Metadata = ()>,
    {
        let raw = Self::into_raw(self) as *mut U;
        GcBox {
            ptr: unsafe { NonNull::new_unchecked(raw as *mut ()) },
            _marker: PhantomData,
        }
    }
}

impl GcBox<dyn Any> {
    /// Attempts to safely cast a GcBox<dyn Any> to GcBox<T>
    pub fn downcast<T>(self) -> Option<GcBox<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &dyn Any = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        let meta = Metadata::new::<T>(concrete);
        let boxed = GcBox {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut ()) },
            _marker: PhantomData,
        };
        let header = header(boxed.ptr.as_ptr());
        unsafe {
            *header = meta;
        }
        Some(boxed)
    }
}

impl GcBox<dyn Any + Send> {
    /// Attempts to safely cast a GcBox<dyn Any> to GcBox<T>
    pub fn downcast<T>(self) -> Option<GcBox<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &(dyn Any + Send) = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        let meta = Metadata::new::<T>(concrete);
        let boxed = GcBox {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut ()) },
            _marker: PhantomData,
        };
        let header = header(boxed.ptr.as_ptr());
        unsafe {
            *header = meta;
        }
        Some(boxed)
    }
}

impl GcBox<dyn Any + Send + Sync> {
    /// Attempts to safely cast a GcBox<dyn Any> to GcBox<T>
    pub fn downcast<T>(self) -> Option<GcBox<T>>
    where
        T: Any,
        PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
    {
        let raw = Self::into_raw(self);
        let any: &(dyn Any + Send + Sync) = unsafe { &*raw };
        let Some(concrete) = any.downcast_ref::<T>() else { return None; };
        let meta = Metadata::new::<T>(concrete);
        let boxed = GcBox {
            ptr: unsafe { NonNull::new_unchecked(concrete as *const _ as *mut ()) },
            _marker: PhantomData,
        };
        let header = header(boxed.ptr.as_ptr());
        unsafe {
            *header = meta;
        }
        Some(boxed)
    }
}

impl<T> GcBox<T>
where
    T: ?Sized + 'static + Pointee<Metadata = usize>,
{
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_capacity_in(cap, Global).unwrap()
    }

    pub fn with_capacity_in<A: Allocator>(cap: usize, alloc: A) -> Result<Self, AllocError> {
        let empty = ptr::from_raw_parts::<T>(ptr::null() as *const (), cap);
        let meta = Metadata::new::<T>(empty);
        let value_layout = unsafe { Layout::for_value_raw(empty) };
        let (layout, value_offset) = Layout::new::<Metadata>().extend(value_layout).unwrap();
        let ptr: NonNull<()> = alloc.allocate(layout)?.cast();
        unsafe {
            let ptr = NonNull::new_unchecked(ptr.as_ptr().byte_add(value_offset));
            let boxed = Self {
                ptr,
                _marker: PhantomData,
            };
            ptr::write(header(boxed.ptr.as_ptr()), meta);
            Ok(boxed)
        }
    }
}

impl<T> GcBox<[T]> {
    #[inline]
    pub fn new_uninit_slice(len: usize) -> GcBox<[MaybeUninit<T>]> {
        GcBox::new_uninit_slice_in(len, Global).unwrap()
    }

    #[inline]
    pub fn new_uninit_slice_in<A: Allocator>(
        len: usize,
        alloc: A,
    ) -> Result<GcBox<[MaybeUninit<T>]>, AllocError> {
        GcBox::<[MaybeUninit<T>]>::with_capacity_in(len, alloc)
    }
}

impl<T: Clone> GcBox<[T]> {
    pub fn to_boxed_slice_in<A: Allocator>(slice: &[T], alloc: A) -> Result<Self, AllocError> {
        let mut boxed = GcBox::<[MaybeUninit<T>]>::with_capacity_in(slice.len(), alloc)?;
        boxed.write_slice_cloned(slice);
        Ok(unsafe { boxed.assume_init() })
    }
}

impl<T: Any> GcBox<MaybeUninit<T>> {
    pub unsafe fn assume_init(self) -> GcBox<T> {
        let raw = GcBox::into_raw(self);
        GcBox::from_raw(raw as *mut T)
    }

    pub fn write(mut boxed: Self, value: T) -> GcBox<T> {
        unsafe {
            (*boxed).write(value);
            boxed.assume_init()
        }
    }
}

impl<T> GcBox<[MaybeUninit<T>]> {
    pub unsafe fn assume_init(self) -> GcBox<[T]> {
        let raw = GcBox::into_raw(self);
        GcBox::from_raw(raw as *mut [T])
    }
}

impl<T: Copy> GcBox<[MaybeUninit<T>]> {
    pub fn write_slice(&mut self, src: &[T]) {
        MaybeUninit::write_slice(&mut **self, src);
    }
}

impl<T: Clone> GcBox<[MaybeUninit<T>]> {
    pub fn write_slice_cloned(&mut self, src: &[T]) {
        MaybeUninit::write_slice_cloned(&mut **self, src);
    }
}

impl<T> Debug for GcBox<T>
where
    T: ?Sized + 'static + Debug,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T> Display for GcBox<T>
where
    T: ?Sized + 'static + Display,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T> fmt::Pointer for GcBox<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<T> borrow::Borrow<T> for GcBox<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T> borrow::BorrowMut<T> for GcBox<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T> AsRef<T> for GcBox<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T> AsMut<T> for GcBox<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T> PartialEq for GcBox<T>
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

impl<T> PartialEq<T> for GcBox<T>
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

impl<T> PartialOrd<T> for GcBox<T>
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

impl<T> PartialOrd for GcBox<T>
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

impl<T> Ord for GcBox<T>
where
    T: ?Sized + 'static + Ord,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T> Eq for GcBox<T>
where
    T: ?Sized + 'static + Eq,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
}

impl<T> Hash for GcBox<T>
where
    T: ?Sized + 'static + Hash,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T> From<T> for GcBox<T>
where
    T: Any,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn from(t: T) -> Self {
        GcBox::new(t)
    }
}

impl<T: Copy> From<&[T]> for GcBox<[T]> {
    fn from(slice: &[T]) -> Self {
        let mut boxed = GcBox::<[MaybeUninit<T>]>::with_capacity(slice.len());
        boxed.write_slice(slice);
        unsafe { boxed.assume_init() }
    }
}

impl<T: Copy> From<Cow<'_, [T]>> for GcBox<[T]> {
    #[inline]
    fn from(cow: Cow<'_, [T]>) -> Self {
        match cow {
            Cow::Borrowed(slice) => GcBox::from(slice),
            Cow::Owned(slice) => GcBox::from(slice.as_slice()),
        }
    }
}

impl<T> Deref for GcBox<T>
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

impl<T> DerefMut for GcBox<T>
where
    T: ?Sized + 'static,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr = self.value();
        unsafe { &mut *ptr }
    }
}

impl<Args, F: FnOnce<Args>> FnOnce<Args> for GcBox<F>
where
    F: 'static,
    PtrMetadata: From<<F as Pointee>::Metadata> + TryInto<<F as Pointee>::Metadata>,
{
    type Output = <F as FnOnce<Args>>::Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output {
        let value = unsafe { ptr::read(self.value()) };
        <F as FnOnce<Args>>::call_once(value, args)
    }
}

impl<Args, F: FnMut<Args>> FnMut<Args> for GcBox<F>
where
    F: 'static,
    PtrMetadata: From<<F as Pointee>::Metadata> + TryInto<<F as Pointee>::Metadata>,
{
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output {
        <F as FnMut<Args>>::call_mut(&mut **self, args)
    }
}

impl<Args, F: Fn<Args>> Fn<Args> for GcBox<F>
where
    F: 'static,
    PtrMetadata: From<<F as Pointee>::Metadata> + TryInto<<F as Pointee>::Metadata>,
{
    extern "rust-call" fn call(&self, args: Args) -> Self::Output {
        <F as Fn<Args>>::call(&**self, args)
    }
}

fn header(ptr: *mut ()) -> *mut Metadata {
    unsafe { ptr.byte_sub(mem::size_of::<Metadata>()).cast() }
}

/// This is a marker type used to indicate that the data pointed to by a GcBox has been
/// forwarded to a new location in memory. The metadata for the GcBox contains the new
/// location
pub struct ForwardingMarker;
impl ForwardingMarker {
    pub const TYPE_ID: TypeId = TypeId::of::<ForwardingMarker>();
}

/// This metadata provides enough information to restore a fat pointer from a thin
/// pointer, and to cast to and from Opaque
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Metadata {
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
            ty,
            meta: meta.into(),
        }
    }

    /// Creates metadata representing forwarding a pointer to a new location in memory
    #[allow(dead_code)]
    fn forward(forwarded: *const ()) -> Self {
        Self {
            ty: ForwardingMarker::TYPE_ID,
            meta: PtrMetadata { forwarded },
        }
    }

    /// Returns true if this metadata indicates the containing GcBox has been forwarded
    #[allow(dead_code)]
    #[inline(always)]
    fn is_forwarded(&self) -> bool {
        self.ty == ForwardingMarker::TYPE_ID
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
    fn is<T: ?Sized + 'static>(&self) -> bool {
        self.ty == TypeId::of::<T>()
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub union PtrMetadata {
    #[allow(dead_code)]
    forwarded: *const (),
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

    struct Closure<Args>(extern "rust-call" fn(Args) -> Option<usize>);
    impl<Args> FnOnce<Args> for Closure<Args> {
        type Output = Option<usize>;
        extern "rust-call" fn call_once(self, args: Args) -> Self::Output {
            (self.0)(args)
        }
    }
    impl<Args> FnMut<Args> for Closure<Args> {
        extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output {
            (self.0)(args)
        }
    }
    impl<Args> Fn<Args> for Closure<Args> {
        extern "rust-call" fn call(&self, args: Args) -> Self::Output {
            (self.0)(args)
        }
    }

    extern "rust-call" fn unboxed_closure(args: (GcBox<Tuple>, usize)) -> Option<usize> {
        let tuple = args.0;
        let index = args.1;
        tuple.get(index)
    }

    #[test]
    fn can_gcbox_pods() {
        let value = GcBox::new(Cons { head: 1, tail: 2 });
        assert_eq!(value.head, 1);
        assert_eq!(value.tail, 2);
    }

    #[test]
    fn can_gcbox_dsts() {
        let mut value = GcBox::<Tuple>::with_capacity(2);
        value.set_element(0, 42);
        value.set_element(1, 11);
        assert_eq!(value.len(), 2);
        assert_eq!(value.get_element(0), 42);
        assert_eq!(value.get_element(1), 11);
    }

    #[test]
    fn can_gcbox_downcast_any() {
        let value = GcBox::<dyn Any>::new_unsize(Cons { head: 1, tail: 2 });
        assert!(value.is::<Cons>());
        let result = value.downcast::<Cons>();
        assert!(result.is_some());
        let cons = result.unwrap();
        assert_eq!(cons.head, 1);
        assert_eq!(cons.tail, 2);
    }

    #[test]
    fn cant_gcbox_downcast_any_improperly() {
        let value = GcBox::<dyn Any>::new_unsize(Cons { head: 1, tail: 2 });
        assert!(!value.is::<usize>());
        let result = value.downcast::<usize>();
        assert!(result.is_none());
    }

    #[test]
    fn can_gcbox_unboxed_closures() {
        let mut tuple = GcBox::<Tuple>::with_capacity(1);
        tuple.set_element(0, 42);
        let closure = GcBox::new(Closure(unboxed_closure));
        let result = closure(tuple, 0);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn can_gcbox_closures() {
        let mut tuple = GcBox::<Tuple>::with_capacity(1);
        tuple.set_element(0, 42);
        let closure = GcBox::new(|tuple: GcBox<Tuple>, index| tuple.get(index));
        let result = closure(tuple, 0);
        assert_eq!(result, Some(42));
    }
}

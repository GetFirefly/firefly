use core::alloc::Layout;
use core::convert::TryFrom;
use core::fmt::{self, Debug};
use core::iter;
use core::ptr::{self, NonNull};
use core::slice;
use core::str;
use core::sync::atomic::{self, AtomicUsize};

use intrusive_collections::LinkedListLink;
use liblumen_core::offset_of;

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::Process;
use crate::erts::string::Encoding;
use crate::erts::term::prelude::*;

/// This is the header written alongside all procbin binaries in the heap,
/// it owns the refcount and the raw binary data
///
/// NOTE: It is critical that if you add fields to this struct, that you adjust
/// the implementation of `base_layout` and `ProcBin::from_slice`, as they must
/// manually calculate the data layout due to the fact that `ProcBinInner` is a
/// dynamically-sized type
#[repr(C)]
pub struct ProcBinInner {
    refc: AtomicUsize,
    flags: BinaryFlags,
    data: [u8],
}
impl_static_header!(ProcBin, Term::HEADER_PROCBIN);
impl ProcBinInner {
    /// Constructs a reference to a `ProcBinInner` given a pointer to
    /// the memory containing the struct and the length of its variable-length
    /// data
    ///
    /// NOTE: For more information about how this works, see the detailed
    /// explanation in the function docs for `HeapBin::from_raw_parts`
    #[inline]
    fn from_raw_parts(ptr: *const u8, len: usize) -> Boxed<Self> {
        // Invariants of slice::from_raw_parts.
        assert!(!ptr.is_null());
        assert!(len <= isize::max_value() as usize);

        unsafe {
            let slice = core::slice::from_raw_parts(ptr as *const (), len);
            Boxed::new_unchecked(slice as *const [()] as *mut Self)
        }
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Produces the base layout for this struct, before the
    /// dynamically sized data is factored in.
    ///
    /// Returns the base layout + the offset of the flags field
    #[inline]
    fn base_layout() -> (Layout, usize) {
        Layout::new::<AtomicUsize>()
            .extend(Layout::new::<BinaryFlags>())
            .unwrap()
    }
}
impl Bitstring for ProcBinInner {
    #[inline]
    fn full_byte_len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    unsafe fn as_byte_ptr(&self) -> *mut u8 {
        self.data.as_ptr() as *mut u8
    }
}
impl Binary for ProcBinInner {
    #[inline]
    fn flags(&self) -> &BinaryFlags {
        &self.flags
    }
}
impl IndexByte for ProcBinInner {
    fn byte(&self, index: usize) -> u8 {
        self.data[index]
    }
}
impl Debug for ProcBinInner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ptr = unsafe { self.as_byte_ptr() };
        let len = self.data.len();
        f.debug_struct("ProcBinInner")
            .field("refc", &self.refc)
            .field("flags", &self.flags)
            .field("data", &format!("bytes={},address={:p}", len, ptr))
            .finish()
    }
}

/// Reference-counted heap-allocated binary
///
/// This struct doesn't actually have the data, but it is the entry point
/// through which all consumers will access it, which ensures the reference
/// count is maintained correctly
#[derive(Debug)]
#[repr(C)]
pub struct ProcBin {
    header: Header<ProcBin>,
    inner: NonNull<ProcBinInner>,
    pub link: LinkedListLink,
}
impl ProcBin {
    #[inline]
    pub fn inner_offset() -> usize {
        offset_of!(ProcBin, inner)
    }

    /// Creates a new procbin from a str slice, by copying it to the heap
    pub fn from_str(s: &str) -> AllocResult<Self> {
        let encoding = Encoding::from_str(s);

        Self::from_slice(s.as_bytes(), encoding)
    }

    /// Creates a new procbin from a raw byte slice, by copying it to the heap
    pub fn from_slice(s: &[u8], encoding: Encoding) -> AllocResult<Self> {
        use liblumen_core::sys::alloc as sys_alloc;

        let (base_layout, flags_offset) = ProcBinInner::base_layout();
        let (unpadded_layout, data_offset) = base_layout.extend(Layout::for_value(s)).unwrap();
        // We pad to alignment so that the Layout produced here
        // matches that returned by `Layout::for_value` on the
        // final `ProcBinInner`
        let layout = unpadded_layout.pad_to_align();

        unsafe {
            let block = sys_alloc::alloc(layout)?;
            let len = s.len();

            let ptr: *mut u8 = block.ptr.as_ptr();
            ptr::write(ptr as *mut AtomicUsize, AtomicUsize::new(1));
            let flags_ptr = ptr.offset(flags_offset as isize) as *mut BinaryFlags;
            let flags = BinaryFlags::new(encoding).set_size(len);
            ptr::write(flags_ptr, flags);
            let data_ptr = ptr.offset(data_offset as isize);
            ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, len);

            let inner = ProcBinInner::from_raw_parts(ptr, len);
            Ok(Self {
                header: Default::default(),
                inner: inner.into(),
                link: LinkedListLink::new(),
            })
        }
    }

    #[inline]
    fn inner(&self) -> &ProcBinInner {
        unsafe { self.inner.as_ref() }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&self) {
        use liblumen_core::sys::alloc as sys_alloc;

        if self.inner().refc.fetch_sub(1, atomic::Ordering::Release) == 1 {
            atomic::fence(atomic::Ordering::Acquire);
            let inner = self.inner.as_ref();
            let layout = Layout::for_value(&inner);
            sys_alloc::free(inner as *const _ as *mut u8, layout);
        }
    }

    #[inline]
    pub fn full_byte_iter<'a>(&'a self) -> iter::Copied<slice::Iter<'a, u8>> {
        self.inner().as_bytes().iter().copied()
    }
}
impl Bitstring for ProcBin {
    #[inline]
    fn full_byte_len(&self) -> usize {
        self.inner().full_byte_len()
    }

    #[inline]
    unsafe fn as_byte_ptr(&self) -> *mut u8 {
        self.inner().as_byte_ptr()
    }
}
impl Binary for ProcBin {
    #[inline]
    fn flags(&self) -> &BinaryFlags {
        self.inner().flags()
    }
}
impl AlignedBinary for ProcBin {
    fn as_bytes(&self) -> &[u8] {
        self.inner().as_bytes()
    }
}

impl Clone for ProcBin {
    #[inline]
    fn clone(&self) -> Self {
        self.inner().refc.fetch_add(1, atomic::Ordering::AcqRel);

        Self {
            header: self.header.clone(),
            inner: self.inner,
            link: LinkedListLink::new(),
        }
    }
}

impl CloneToProcess for ProcBin {
    fn clone_to_process(&self, process: &Process) -> Term {
        let mut heap = process.acquire_heap();
        let boxed = self.clone_to_heap(&mut heap).unwrap();
        let ptr: *mut Self = boxed.dyn_cast();
        self.inner().refc.fetch_add(1, atomic::Ordering::AcqRel);
        // Reify a reference to the newly written clone, and push it
        // on to the process virtual heap
        let clone = unsafe { &*ptr };
        process.virtual_alloc(clone);
        boxed
    }

    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
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
                    link: LinkedListLink::new(),
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

impl Drop for ProcBin {
    fn drop(&mut self) {
        if self.inner().refc.fetch_sub(1, atomic::Ordering::Release) != 1 {
            return;
        }
        // The following code is based on the Rust Arc<T> implementation, and
        // their notes apply to us here:
        //
        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data.  Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // In particular, while the contents of an Arc are usually immutable, it's
        // possible to have interior writes to something like a Mutex<T>. Since a
        // Mutex is not acquired when it is deleted, we can't rely on its
        // synchronization logic to make writes in thread A visible to a destructor
        // running in thread B.
        //
        // Also note that the Acquire fence here could probably be replaced with an
        // Acquire load, which could improve performance in highly-contended
        // situations. See [2].
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // [2]: (https://github.com/rust-lang/rust/pull/41714)
        atomic::fence(atomic::Ordering::Acquire);
        // The refcount is now zero, so we are freeing the memory
        unsafe {
            self.drop_slow();
        }
    }
}

impl IndexByte for ProcBin {
    fn byte(&self, index: usize) -> u8 {
        self.inner().byte(index)
    }
}

/// Given a raw pointer to the ProcBin, reborrows and clones it into a new reference.
///
/// # Safety
///
/// This function is unsafe due to dereferencing a raw pointer, but it is expected that
/// this is only ever called with a valid `ProcBin` pointer anyway. The primary risk
/// with obtaining a `ProcBin` via this function is if you leak it somehow, rather than
/// letting its `Drop` implementation run. Doing so will leave the reference count greater
/// than 1 forever, meaning memory will never get deallocated.
///
/// NOTE: This does not copy the binary, it only obtains a new `ProcBin`, which is
/// itself a reference to a binary held by a `ProcBinInner`.
impl From<Boxed<ProcBin>> for ProcBin {
    fn from(boxed: Boxed<ProcBin>) -> Self {
        boxed.as_ref().clone()
    }
}
impl TryFrom<TypedTerm> for Boxed<ProcBin> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::ProcBin(term) => Ok(term),
            _ => Err(TypeError),
        }
    }
}

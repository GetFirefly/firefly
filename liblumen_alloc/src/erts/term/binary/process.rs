use core::alloc::Layout;
use core::cmp;
use core::convert::TryInto;
use core::fmt::{self, Debug};
use core::mem;
use core::ptr::{self, NonNull};
use core::slice;
use core::str;
use core::sync::atomic::{self, AtomicUsize};

use alloc::borrow::ToOwned;
use alloc::string::String;

use intrusive_collections::LinkedListLink;

use crate::borrow::CloneToProcess;
use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
use crate::erts::process::Process;
use crate::erts::term::binary::heap::HeapBin;
use crate::erts::term::binary::sub::{Original, SubBinary};
use crate::erts::term::{arity_of, AsTerm, Boxed, MatchContext, Term};
use crate::erts::HeapAlloc;

use super::{
    aligned_binary::AlignedBinary, BinaryType, Bitstring, FLAG_IS_LATIN1_BIN, FLAG_IS_RAW_BIN,
    FLAG_IS_UTF8_BIN, FLAG_MASK,
};

/// This is the header written alongside all procbin binaries in the heap,
/// it owns the refcount and has the pointer to the data and its size
#[repr(C)]
pub struct ProcBinInner {
    refc: AtomicUsize,
    flags: usize,
    bytes: *mut u8,
}
impl ProcBinInner {
    #[inline]
    fn bytes(&self) -> *mut u8 {
        self.bytes
    }

    #[inline]
    fn full_byte_len(&self) -> usize {
        self.flags & !FLAG_MASK
    }

    #[inline]
    fn binary_type(&self) -> BinaryType {
        BinaryType::from_flags(self.flags)
    }

    /// Returns true if this binary is a raw binary
    #[inline]
    fn is_raw(&self) -> bool {
        self.flags & FLAG_MASK == FLAG_IS_RAW_BIN
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    fn is_latin1(&self) -> bool {
        self.flags & FLAG_MASK == FLAG_IS_LATIN1_BIN
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    fn is_utf8(&self) -> bool {
        self.flags & FLAG_MASK == FLAG_IS_UTF8_BIN
    }
}

impl Debug for ProcBinInner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ProcBinInner")
            .field("refc", &self.refc)
            .field("flags", &format_args!("{:#b}", self.flags))
            .field("bytes", &self.bytes)
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
    pub(super) header: Term,
    inner: NonNull<ProcBinInner>,
    pub link: LinkedListLink,
}
impl ProcBin {
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
    pub unsafe fn from_raw(term: *mut ProcBin) -> Self {
        let bin = &*term;
        bin.clone()
    }

    /// Converts `self` into a raw pointer
    ///
    /// # Safety
    ///
    /// This operation leaks `self` and skips decrementing the reference count,
    /// so it is essential that the resulting pointer is dereferenced at least once
    /// directly, or by using `from_raw_noincrement`, in order to ensure that the
    /// reference count is properly decremented, and the underlying memory is properly
    /// freed.
    ///
    /// This function is intended for internal use only, specifically when initially
    /// creating a `ProcBin` and creating a boxed `Term` from the resulting pointer.
    /// This is the only time where we want to avoid dropping the value and decrementing
    /// the reference count, since the `ProcBin` is still referenced, it has just been lowered
    /// to a "primitive" value and placed on a process heap.
    #[allow(unused)]
    pub(crate) unsafe fn into_raw(self) -> *mut ProcBin {
        let ptr = &self as *const _ as *mut _;
        mem::forget(self);
        ptr
    }

    /// Returns true if this binary is a raw binary
    #[inline]
    pub fn is_raw(&self) -> bool {
        self.inner().is_raw()
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.inner().is_latin1()
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    pub fn is_utf8(&self) -> bool {
        self.inner().is_utf8()
    }

    /// Returns a `BinaryType` representing the encoding type of this binary
    #[inline]
    pub fn binary_type(&self) -> BinaryType {
        self.inner().binary_type()
    }

    /// Returns a raw pointer to the binary data underlying this `ProcBin`
    ///
    /// # Safety
    ///
    /// This is only intended for use by garbage collection, in order to
    /// update match context references. You should use `as_bytes` instead,
    /// as it produces a byte slice which is safe to work with, whereas the
    /// pointer returned here is not
    #[inline]
    pub(crate) fn bytes(&self) -> *mut u8 {
        self.inner().bytes()
    }

    /// Creates a new procbin from a str slice, by copying it to the heap
    pub fn from_str(s: &str) -> Result<Self, Alloc> {
        let binary_type = BinaryType::from_str(s);

        Self::from_slice(s.as_bytes(), binary_type)
    }

    /// Creates a new procbin from a raw byte slice, by copying it to the heap
    pub fn from_slice(s: &[u8], binary_type: BinaryType) -> Result<Self, Alloc> {
        use liblumen_core::sys::alloc as sys_alloc;

        let full_byte_len = s.len();
        let (layout, offset) = Layout::new::<ProcBinInner>()
            .extend(unsafe {
                Layout::from_size_align_unchecked(full_byte_len, mem::align_of::<u8>())
            })
            .unwrap();

        unsafe {
            match sys_alloc::alloc(layout) {
                Ok(non_null) => {
                    let ptr = non_null.as_ptr();
                    let inner_ptr = ptr as *mut ProcBinInner;
                    let bytes = ptr.add(offset);

                    inner_ptr.write(ProcBinInner {
                        refc: AtomicUsize::new(1),
                        flags: full_byte_len | binary_type.to_flags(),
                        bytes,
                    });
                    ptr::copy_nonoverlapping(s.as_ptr(), bytes, full_byte_len);

                    Ok(Self {
                        header: Term::make_header(arity_of::<Self>(), Term::FLAG_PROCBIN),
                        inner: NonNull::new_unchecked(inner_ptr),
                        link: LinkedListLink::new(),
                    })
                }
                Err(_) => Err(alloc!()),
            }
        }
    }

    /// Converts this binary to a `&str` slice.
    ///
    /// This conversion does not move the string, it can be considered as
    /// creating a new reference with a lifetime attached to that of `self`.
    ///
    /// Due to the fact that the lifetime of the `&str` cannot outlive the
    /// `ProcBin`, this does not require incrementing the reference count.
    #[allow(unused)]
    pub fn as_str<'a>(&'a self) -> &'a str {
        assert!(
            self.is_latin1() || self.is_utf8(),
            "cannot convert a binary containing non-UTF-8/non-ASCII characters to &str"
        );
        unsafe {
            let bytes = self.as_bytes();
            str::from_utf8_unchecked(bytes)
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

        // Destroy the data at this time, even though we may not free the box
        // allocation itself (there may still be weak pointers lying around).

        if self.inner().refc.fetch_sub(1, atomic::Ordering::Release) == 1 {
            atomic::fence(atomic::Ordering::Acquire);
            let bytes = self.inner().bytes();
            let size = self.inner().full_byte_len();
            sys_alloc::free(
                bytes,
                Layout::from_size_align_unchecked(size, mem::align_of::<usize>()),
            );
        }
    }
}

unsafe impl AsTerm for ProcBin {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl AlignedBinary for ProcBin {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            let inner = self.inner();
            slice::from_raw_parts(inner.bytes(), inner.full_byte_len())
        }
    }
}

impl Bitstring for ProcBin {
    fn full_byte_len(&self) -> usize {
        self.inner().full_byte_len()
    }
}

impl Clone for ProcBin {
    #[inline]
    fn clone(&self) -> Self {
        self.inner().refc.fetch_add(1, atomic::Ordering::AcqRel);

        Self {
            header: self.header,
            inner: self.inner,
            link: LinkedListLink::new(),
        }
    }
}

impl CloneToProcess for ProcBin {
    fn clone_to_process(&self, process: &Process) -> Term {
        let mut heap = process.acquire_heap();
        let boxed = self.clone_to_heap(&mut heap).unwrap();
        let ptr = boxed.boxed_val() as *mut Self;
        self.inner().refc.fetch_add(1, atomic::Ordering::AcqRel);
        // Reify a reference to the newly written clone, and push it
        // on to the process virtual heap
        let clone = unsafe { &*ptr };
        process.virtual_alloc(clone);
        boxed
    }

    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        unsafe {
            // Allocate space for the header
            let layout = Layout::new::<Self>();
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            // Write the binary header with an empty link
            ptr::write(
                ptr,
                Self {
                    header: self.header,
                    inner: self.inner,
                    link: LinkedListLink::new(),
                },
            );
            // Reify result term
            Ok(Term::make_boxed(ptr))
        }
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

impl Eq for ProcBin {}

impl Original for ProcBin {
    fn byte(&self, index: usize) -> u8 {
        let inner = self.inner();
        let full_byte_len = inner.full_byte_len();

        assert!(
            index < full_byte_len,
            "index ({}) >= full_byte_len ({})",
            index,
            full_byte_len
        );

        unsafe { *inner.bytes().add(index) }
    }
}

impl PartialEq<Boxed<HeapBin>> for ProcBin {
    fn eq(&self, other: &Boxed<HeapBin>) -> bool {
        self.eq(other.as_ref())
    }
}

impl PartialEq<MatchContext> for ProcBin {
    fn eq(&self, other: &MatchContext) -> bool {
        other.eq(self)
    }
}

impl PartialEq<SubBinary> for ProcBin {
    fn eq(&self, other: &SubBinary) -> bool {
        other.eq(self)
    }
}

impl PartialOrd<HeapBin> for ProcBin {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn partial_cmp(&self, other: &HeapBin) -> Option<core::cmp::Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

impl PartialOrd<Boxed<HeapBin>> for ProcBin {
    fn partial_cmp(&self, other: &Boxed<HeapBin>) -> Option<cmp::Ordering> {
        self.partial_cmp(other.as_ref())
    }
}

impl PartialOrd<MatchContext> for ProcBin {
    fn partial_cmp(&self, other: &MatchContext) -> Option<cmp::Ordering> {
        other.partial_cmp(self).map(|ordering| ordering.reverse())
    }
}

impl PartialOrd<SubBinary> for ProcBin {
    fn partial_cmp(&self, other: &SubBinary) -> Option<cmp::Ordering> {
        other.partial_cmp(self).map(|ordering| ordering.reverse())
    }
}

impl TryInto<String> for ProcBin {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        match str::from_utf8(self.as_bytes()) {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => Err(badarg!()),
        }
    }
}

use core::alloc::Layout;
use core::cmp;
use core::fmt;
use core::mem;
use core::ptr::{self, NonNull};
use core::slice;
use core::str;
use core::sync::atomic;
use core::sync::atomic::{AtomicUsize, Ordering};

use intrusive_collections::LinkedListLink;
use liblumen_core::util::pointer::distance_absolute;

use crate::borrow::CloneToProcess;

use super::*;

pub trait Binary {
    fn as_bytes(&self) -> &[u8];
}

const FLAG_SHIFT: usize = mem::size_of::<usize>() * 8 - 2;
const FLAG_IS_RAW_BIN: usize = 1 << FLAG_SHIFT;
const FLAG_IS_LATIN1_BIN: usize = 2 << FLAG_SHIFT;
const FLAG_IS_UTF8_BIN: usize = 3 << FLAG_SHIFT;
const FLAG_MASK: usize = FLAG_IS_RAW_BIN | FLAG_IS_LATIN1_BIN | FLAG_IS_UTF8_BIN;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BinaryType {
    Raw,
    Latin1,
    Utf8,
}
impl BinaryType {
    #[inline]
    pub fn to_flags(&self) -> usize {
        match self {
            &BinaryType::Raw => FLAG_IS_RAW_BIN,
            &BinaryType::Latin1 => FLAG_IS_LATIN1_BIN,
            &BinaryType::Utf8 => FLAG_IS_UTF8_BIN,
        }
    }

    #[inline]
    pub fn from_flags(flags: usize) -> Self {
        match flags & FLAG_MASK {
            FLAG_IS_RAW_BIN => BinaryType::Raw,
            FLAG_IS_LATIN1_BIN => BinaryType::Latin1,
            FLAG_IS_UTF8_BIN => BinaryType::Utf8,
            _ => panic!(
                "invalid flags value given to BinaryType::from_flags: {}",
                flags
            ),
        }
    }
}

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
    fn size(&self) -> usize {
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

/// Reference-counted heap-allocated binary
///
/// This struct doesn't actually have the data, but it is the entry point
/// through which all consumers will access it, which ensures the reference
/// count is maintained correctly
#[derive(Debug)]
#[repr(C)]
pub struct ProcBin {
    header: Term,
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

    /// Returns the size of this binary in bytes
    #[inline]
    pub fn size(&self) -> usize {
        self.inner().size()
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
    pub fn from_str(s: &str) -> Result<Self, AllocErr> {
        use liblumen_core::sys::alloc as sys_alloc;

        let size = s.len();
        let (layout, _) = Layout::new::<ProcBinInner>()
            .extend(unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<u8>()) })
            .unwrap();
        let ptr = unsafe { sys_alloc::alloc(layout)?.as_ptr() };
        let header_ptr = ptr as *mut ProcBinInner;
        unsafe {
            // For efficient checks on binary type later, store flags in the pointer
            let bytes = ptr.offset(1) as *mut u8;
            let flags = if s.is_ascii() {
                FLAG_IS_LATIN1_BIN
            } else {
                FLAG_IS_UTF8_BIN
            };
            ptr::write(
                ptr as *mut ProcBinInner,
                ProcBinInner {
                    refc: AtomicUsize::new(1),
                    flags: size | flags,
                    bytes,
                },
            );
            ptr::copy_nonoverlapping(s.as_ptr(), bytes, size);

            let arityval = to_word_size(mem::size_of::<Self>() - mem::size_of::<Term>());
            Ok(Self {
                header: Term::make_header(arityval, Term::FLAG_PROCBIN),
                inner: NonNull::new_unchecked(header_ptr),
                link: LinkedListLink::new(),
            })
        }
    }

    /// Creates a new procbin from a raw byte slice, by copying it to the heap
    pub fn from_slice(s: &[u8]) -> Result<Self, AllocErr> {
        use liblumen_core::sys::alloc as sys_alloc;

        let size = s.len();
        let (layout, _) = Layout::new::<ProcBinInner>()
            .extend(unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<u8>()) })
            .unwrap();
        let ptr = unsafe { sys_alloc::alloc(layout)?.as_ptr() };
        let header_ptr = ptr as *mut ProcBinInner;
        unsafe {
            // For efficient checks on binary type later, store flags in the pointer
            let bytes = ptr.offset(1) as *mut u8;
            ptr::write(
                ptr as *mut ProcBinInner,
                ProcBinInner {
                    refc: AtomicUsize::new(1),
                    flags: size | FLAG_IS_RAW_BIN,
                    bytes,
                },
            );
            ptr::copy_nonoverlapping(s.as_ptr(), bytes, size);

            let arityval = to_word_size(mem::size_of::<Self>() - mem::size_of::<Term>());
            Ok(Self {
                header: Term::make_header(arityval, Term::FLAG_PROCBIN),
                inner: NonNull::new_unchecked(header_ptr),
                link: LinkedListLink::new(),
            })
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
            let inner = self.inner();
            let bytes = slice::from_raw_parts(inner.bytes(), inner.size());
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

        if self.inner().refc.fetch_sub(1, Ordering::Release) == 1 {
            atomic::fence(Ordering::Acquire);
            let bytes = self.inner().bytes();
            let size = self.inner().size();
            sys_alloc::free(
                bytes,
                Layout::from_size_align_unchecked(size, mem::align_of::<usize>()),
            );
        }
    }
}
impl Clone for ProcBin {
    #[inline]
    fn clone(&self) -> Self {
        self.inner().refc.fetch_add(1, Ordering::AcqRel);

        Self {
            header: self.header,
            inner: self.inner,
            link: LinkedListLink::new(),
        }
    }
}

impl Binary for ProcBin {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            let inner = self.inner();
            slice::from_raw_parts(inner.bytes(), inner.size())
        }
    }
}

impl Drop for ProcBin {
    fn drop(&mut self) {
        if self.inner().refc.fetch_sub(1, Ordering::Release) != 1 {
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
        atomic::fence(Ordering::Acquire);
        // The refcount is now zero, so we are freeing the memory
        unsafe {
            self.drop_slow();
        }
    }
}

impl<B: Binary> PartialEq<B> for ProcBin {
    fn eq(&self, other: &B) -> bool {
        self.as_bytes().eq(other.as_bytes())
    }
}
impl Eq for ProcBin {}
impl<B: Binary> PartialOrd<B> for ProcBin {
    fn partial_cmp(&self, other: &B) -> Option<cmp::Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

unsafe impl AsTerm for ProcBin {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl CloneToProcess for ProcBin {
    fn clone_to_process<A: AllocInProcess>(&self, process: &mut A) -> Term {
        self.inner().refc.fetch_add(1, Ordering::AcqRel);
        unsafe {
            // Allocate space for the header
            let layout = Layout::new::<Self>();
            let ptr = process.alloc_layout(layout).unwrap().as_ptr() as *mut Self;
            // Write the binary header with an empty link
            ptr::write(
                ptr,
                Self {
                    header: self.header,
                    inner: self.inner,
                    link: LinkedListLink::new(),
                },
            );
            // Reify a reference to the newly written clone, and push it
            // on to the process virtual heap
            let clone = &*ptr;
            process.virtual_alloc(clone);
            // Reify result term
            Term::make_boxed(ptr)
        }
    }
}

/// Process heap allocated binary, smaller than 64 bytes
#[derive(Debug, Clone)]
#[repr(C)]
pub struct HeapBin {
    header: Term,
    flags: usize,
}

impl HeapBin {
    // The size of the extra fields in bytes
    const EXTRA_ARITYVAL: usize = mem::size_of::<Self>() - mem::size_of::<usize>();

    /// Create a new `HeapBin` header which will point to a binary of size `size`
    #[inline]
    pub fn new(size: usize) -> Self {
        let words = to_word_size(size) + to_word_size(Self::EXTRA_ARITYVAL);
        Self {
            header: Term::make_header(words, Term::FLAG_HEAPBIN),
            flags: size | FLAG_IS_RAW_BIN,
        }
    }

    /// Like `new`, but for latin1-encoded binaries
    #[inline]
    pub fn new_latin1(size: usize) -> Self {
        let words = to_word_size(size) + to_word_size(Self::EXTRA_ARITYVAL);
        Self {
            header: Term::make_header(words, Term::FLAG_HEAPBIN),
            flags: size | FLAG_IS_LATIN1_BIN,
        }
    }
    /// Like `new`, but for utf8-encoded binaries
    #[inline]
    pub fn new_utf8(size: usize) -> Self {
        let words = to_word_size(size) + to_word_size(Self::EXTRA_ARITYVAL);
        Self {
            header: Term::make_header(words, Term::FLAG_HEAPBIN),
            flags: size | FLAG_IS_UTF8_BIN,
        }
    }

    #[inline]
    pub(in crate::erts) fn from_raw_parts(header: Term, flags: usize) -> Self {
        Self { header, flags }
    }

    /// Returns true if this binary is a raw binary
    #[inline]
    pub fn is_raw(&self) -> bool {
        self.flags & FLAG_MASK == FLAG_IS_RAW_BIN
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.flags & FLAG_MASK == FLAG_IS_LATIN1_BIN
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    pub fn is_utf8(&self) -> bool {
        self.flags & FLAG_MASK == FLAG_IS_UTF8_BIN
    }

    /// Returns a `BinaryType` representing the encoding type of this binary
    #[inline]
    pub fn binary_type(&self) -> BinaryType {
        BinaryType::from_flags(self.flags)
    }

    /// Returns the size of just the binary data of this HeapBin in bytes
    #[inline]
    pub fn size(&self) -> usize {
        self.flags & !FLAG_MASK
    }

    /// Returns a raw pointer to the binary data underlying this `HeapBin`
    ///
    /// # Safety
    ///
    /// This is only intended for use by garbage collection, in order to
    /// update match context references. You should use `as_bytes` instead,
    /// as it produces a byte slice which is safe to work with, whereas the
    /// pointer returned here is not
    #[inline]
    pub(crate) fn bytes(&self) -> *mut u8 {
        unsafe { (self as *const Self).offset(1) as *mut u8 }
    }

    /// Get a `Layout` describing the necessary layout to allocate a `HeapBin` for the given string
    #[inline]
    pub fn layout(input: &str) -> Layout {
        let size = mem::size_of::<Self>() + input.len();
        unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<Term>()) }
    }

    /// Get a `Layout` describing the necessary layout to allocate a `HeapBin` for the given byte
    /// slice
    #[inline]
    pub fn layout_bytes(input: &[u8]) -> Layout {
        let size = mem::size_of::<Self>() + input.len();
        unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<Term>()) }
    }

    /// Reifies a `HeapBin` from a raw, untagged, pointer
    #[inline]
    pub unsafe fn from_raw(term: *mut HeapBin) -> Self {
        let hb = &*term;
        hb.clone()
    }

    /// Converts this binary to a `&str` slice.
    ///
    /// This conversion does not move the string, it can be considered as
    /// creating a new reference with a lifetime attached to that of `self`.
    #[inline]
    pub fn as_str<'a>(&'a self) -> &'a str {
        assert!(
            self.is_latin1() || self.is_utf8(),
            "cannot convert a binary containing non-UTF-8/non-ASCII characters to &str"
        );
        unsafe {
            let bytes = slice::from_raw_parts(self.bytes(), self.size());
            str::from_utf8_unchecked(bytes)
        }
    }
}

impl Binary for HeapBin {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.bytes(), self.size()) }
    }
}

impl<B: Binary> PartialEq<B> for HeapBin {
    #[inline]
    fn eq(&self, other: &B) -> bool {
        self.as_bytes().eq(other.as_bytes())
    }
}
impl Eq for HeapBin {}
impl<B: Binary> PartialOrd<B> for HeapBin {
    #[inline]
    fn partial_cmp(&self, other: &B) -> Option<cmp::Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

unsafe impl AsTerm for HeapBin {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl CloneToProcess for HeapBin {
    fn clone_to_process<A: AllocInProcess>(&self, process: &mut A) -> Term {
        let bin_size = self.size();
        let size = mem::size_of::<Self>() + bin_size;
        let words = to_word_size(size);
        unsafe {
            // Allocate space for header + binary
            let ptr = process.alloc(words).unwrap().as_ptr() as *mut Self;
            // Copy header
            ptr::copy_nonoverlapping(self as *const Self, ptr, mem::size_of::<Self>());
            // Copy binary
            let bin_ptr = ptr.offset(1) as *mut u8;
            ptr::copy_nonoverlapping(self.bytes(), bin_ptr, bin_size);
            // Return term
            let hb = &*ptr;
            hb.as_term()
        }
    }
}

/// A slice of a binary
#[derive(Clone, Copy)]
#[repr(C)]
pub struct SubBinary {
    header: Term,
    // Binary size in bytes
    size: usize,
    // Offset into original binary
    offset: usize,
    // Size of binary in bits
    bitsize: usize,
    // Offset in bits
    bitoffs: usize,
    // Indicates the underlying binary is writable
    writable: bool,
    // Original binary term (ProcBin or HeapBin)
    orig: Term,
}
impl SubBinary {
    /// See erts_bs_get_binary_2 in erl_bits.c:460
    #[inline]
    pub fn from_match(ctx: &mut MatchContext, num_bits: usize) -> Self {
        assert!(ctx.buffer.size - ctx.buffer.offset < num_bits);

        let orig = ctx.buffer.orig;
        let arityval = word_size_of::<Self>();
        let header = Term::make_header(arityval, Term::FLAG_SUBBINARY);
        let size = byte_offset(num_bits);
        let bitsize = bit_offset(num_bits);
        let offset = byte_offset(ctx.buffer.offset);
        let bitoffs = bit_offset(ctx.buffer.offset);
        let writable = false;
        ctx.buffer.offset += num_bits;
        Self {
            header,
            size,
            offset,
            bitsize,
            bitoffs,
            writable,
            orig,
        }
    }

    #[inline]
    pub unsafe fn from_raw(ptr: *mut SubBinary) -> Self {
        *ptr
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn bytes(&self) -> *mut u8 {
        let real_bin_ptr = follow_moved(self.orig).boxed_val();
        let real_bin = unsafe { *real_bin_ptr };
        if real_bin.is_procbin() {
            let bin = unsafe { &*(real_bin_ptr as *mut ProcBin) };
            bin.bytes()
        } else {
            assert!(real_bin.is_heapbin());
            let bin = unsafe { &*(real_bin_ptr as *mut HeapBin) };
            bin.bytes()
        }
    }

    /// During garbage collection, we sometimes want to convert sub-binary terms
    /// into full-fledged heap binaries, so that the original full-size binary can be freed.
    ///
    /// If this sub-binary is a candidate for conversion, then it will return `Ok((ptr, size))`,
    /// otherwise it will return `Err(())`. The returned pointer and size is sufficient for
    /// passing to `ptr::copy_nonoverlapping` during creation of the new HeapBin.
    ///
    /// NOTE: You should not use this for any other purpose
    pub(crate) fn to_heapbin_parts(&self) -> Result<(Term, usize, *mut u8, usize), ()> {
        if self.bitsize == 0 && self.bitoffs == 0 && !self.writable && self.size <= 64 {
            Ok(unsafe { self.to_raw_parts() })
        } else {
            Err(())
        }
    }

    #[inline]
    unsafe fn to_raw_parts(&self) -> (Term, usize, *mut u8, usize) {
        let real_bin_ptr = follow_moved(self.orig).boxed_val();
        let real_bin = *real_bin_ptr;
        if real_bin.is_procbin() {
            let bin = &*(real_bin_ptr as *mut ProcBin);
            let bytes = bin.bytes().offset(self.offset as isize);
            let flags = bin.binary_type().to_flags();
            (bin.header, flags, bytes, self.size)
        } else {
            assert!(real_bin.is_heapbin());
            let bin = &*(real_bin_ptr as *mut HeapBin);
            let bytes = bin.bytes().offset(self.offset as isize);
            let flags = bin.binary_type().to_flags();
            (bin.header, flags, bytes, self.size)
        }
    }
}
unsafe impl AsTerm for SubBinary {
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl Binary for SubBinary {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            let (_header, _flags, ptr, size) = self.to_raw_parts();
            slice::from_raw_parts(ptr, size)
        }
    }
}

impl<B: Binary> PartialEq<B> for SubBinary {
    #[inline]
    fn eq(&self, other: &B) -> bool {
        self.as_bytes().eq(other.as_bytes())
    }
}
impl Eq for SubBinary {}
impl<B: Binary> PartialOrd<B> for SubBinary {
    #[inline]
    fn partial_cmp(&self, other: &B) -> Option<cmp::Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

impl CloneToProcess for SubBinary {
    fn clone_to_process<A: AllocInProcess>(&self, process: &mut A) -> Term {
        let real_bin_ptr = follow_moved(self.orig).boxed_val();
        let real_bin = unsafe { *real_bin_ptr };
        // For ref-counted binaries and those that are already on the process heap,
        // we just need to copy the sub binary header, not the binary as well
        if real_bin.is_procbin() || (real_bin.is_heapbin() && process.is_owner(real_bin_ptr)) {
            let layout = Layout::new::<Self>();
            let size = layout.size();
            unsafe {
                // Allocate space for header and copy it
                let ptr = process.alloc_layout(layout).unwrap().as_ptr() as *mut Self;
                ptr::copy_nonoverlapping(self as *const Self, ptr, size);
                let sb = &*ptr;
                sb.as_term()
            }
        } else {
            assert!(real_bin.is_heapbin());
            // Need to make sure that the heapbin is cloned as well, and that the header is suitably
            // updated
            let bin = unsafe { &*(real_bin_ptr as *mut HeapBin) };
            let new_bin = bin.clone_to_process(process);
            let layout = Layout::new::<Self>();
            unsafe {
                // Allocate space for header
                let ptr = process.alloc_layout(layout).unwrap().as_ptr() as *mut Self;
                // Write header, with modifications
                ptr::write(
                    ptr,
                    Self {
                        header: self.header,
                        size: self.size,
                        offset: self.offset,
                        bitsize: self.bitsize,
                        bitoffs: self.bitoffs,
                        writable: self.writable,
                        orig: new_bin,
                    },
                );
                let sb = &*ptr;
                sb.as_term()
            }
        }
    }
}
impl fmt::Debug for SubBinary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SubBinary")
            .field("header", &self.header.as_usize())
            .field("size", &self.size)
            .field("offset", &self.offset)
            .field("bitsize", &self.bitsize)
            .field("bitoffs", &self.bitoffs)
            .field("writable", &self.writable)
            .field("orig", &self.orig)
            .finish()
    }
}

/// Represents a binary being matched
///
/// See `ErlBinMatchBuffer` in `erl_bits.h`
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MatchBuffer {
    // Original binary term
    orig: Term,
    // Current position in binary
    base: *mut u8,
    // Offset in bits
    offset: usize,
    // Size of binary in bits
    size: usize,
}
impl MatchBuffer {
    /// Create a match buffer from a binary
    ///
    /// See `erts_bs_start_match_2` in `erl_bits.c`
    #[inline]
    pub fn start_match(orig: Term) -> Self {
        assert!(orig.is_boxed());

        let bin_ptr = orig.boxed_val();
        let bin = unsafe { *bin_ptr };
        let (base, size, offset, bitoffs, bitsize) = if bin.is_procbin() {
            let pb = unsafe { &*(bin_ptr as *mut ProcBin) };
            (pb.bytes(), pb.size() * 8, 0, 0, 0)
        } else if bin.is_heapbin() {
            let hb = unsafe { &*(bin_ptr as *mut HeapBin) };
            (hb.bytes(), hb.size() * 8, 0, 0, 0)
        } else {
            assert!(bin.is_subbinary());
            let sb = unsafe { &*(bin_ptr as *mut SubBinary) };
            (sb.bytes(), sb.size() * 8, sb.offset, sb.bitoffs, sb.bitsize)
        };
        let offset = 8 * offset + bitoffs;
        Self {
            orig,
            base,
            offset,
            size: size + offset + bitsize,
        }
    }
}

/// Used in match contexts
///
/// See `ErlBinMatchState` and `ErlBinMatchBuffer` in `erl_bits.h`
#[derive(Clone, Copy)]
#[repr(C)]
pub struct MatchContext {
    header: Term,
    buffer: MatchBuffer,
    // Saved offsets for contexts created via `bs_start_match2`
    save_offset: Option<usize>,
}
impl MatchContext {
    /// Create a new MatchContext from a boxed procbin/heapbin/sub-bin
    ///
    /// See `erts_bs_start_match_2` in `erl_bits.c`
    #[inline]
    pub fn new(orig: Term) -> Self {
        let buffer = MatchBuffer::start_match(orig);
        let save_offset = if buffer.offset > 0 {
            Some(buffer.offset)
        } else {
            None
        };
        let arityval = to_word_size(mem::size_of::<Self>() - mem::size_of::<Term>());
        Self {
            header: Term::make_header(arityval, Term::FLAG_MATCH_CTX),
            buffer,
            save_offset,
        }
    }

    #[inline]
    pub unsafe fn from_raw(ptr: *mut MatchContext) -> Self {
        *ptr
    }

    /// Used by garbage collection to get a pointer to the original
    /// term in order to place/modify move markers
    #[inline]
    pub(crate) fn orig(&self) -> *mut Term {
        &self.buffer.orig as *const _ as *mut Term
    }

    /// Used by garbage collection to get a pointer to the raw binary
    /// data pointer in order to update it if the underlying binary moves
    #[inline]
    pub(crate) fn base(&self) -> *mut *mut u8 {
        &self.buffer.base as *const _ as *mut *mut u8
    }

    #[inline]
    unsafe fn to_raw_parts(&self) -> (Term, usize, *mut u8, usize) {
        let real_bin_ptr = follow_moved(self.buffer.orig).boxed_val();
        let real_bin = *real_bin_ptr;
        if real_bin.is_procbin() {
            let bin = &*(real_bin_ptr as *mut ProcBin);
            let bytes = bin.bytes().offset(byte_offset(self.buffer.offset) as isize);
            let flags = bin.binary_type().to_flags();
            (bin.header, flags, bytes, num_bytes(self.buffer.size))
        } else {
            assert!(real_bin.is_heapbin());
            let bin = &*(real_bin_ptr as *mut HeapBin);
            let bytes = bin.bytes().offset(byte_offset(self.buffer.offset) as isize);
            let flags = bin.binary_type().to_flags();
            (bin.header, flags, bytes, num_bytes(self.buffer.size))
        }
    }
}
unsafe impl AsTerm for MatchContext {
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl Binary for MatchContext {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            let (_header, _flags, ptr, size) = self.to_raw_parts();
            slice::from_raw_parts(ptr, size)
        }
    }
}

impl<B: Binary> PartialEq<B> for MatchContext {
    #[inline]
    fn eq(&self, other: &B) -> bool {
        self.as_bytes().eq(other.as_bytes())
    }
}
impl Eq for MatchContext {}
impl<B: Binary> PartialOrd<B> for MatchContext {
    #[inline]
    fn partial_cmp(&self, other: &B) -> Option<cmp::Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

impl CloneToProcess for MatchContext {
    fn clone_to_process<A: AllocInProcess>(&self, process: &mut A) -> Term {
        let real_bin_ptr = follow_moved(self.buffer.orig).boxed_val();
        let real_bin = unsafe { *real_bin_ptr };
        // For ref-counted binaries and those that are already on the process heap,
        // we just need to copy the match context header, not the binary as well
        if real_bin.is_procbin() || (real_bin.is_heapbin() && process.is_owner(real_bin_ptr)) {
            let layout = Layout::new::<Self>();
            let size = layout.size();
            unsafe {
                // Allocate space for header and copy it
                let ptr = process.alloc_layout(layout).unwrap().as_ptr() as *mut Self;
                ptr::copy_nonoverlapping(self as *const Self, ptr, size);
                let mc = &*ptr;
                mc.as_term()
            }
        } else {
            assert!(real_bin.is_heapbin());
            // Need to make sure that the heapbin is cloned as well, and that the header is suitably
            // updated
            let bin = unsafe { &*(real_bin_ptr as *mut HeapBin) };
            let new_bin = bin.clone_to_process(process);
            let new_bin_ref = unsafe { &*(new_bin.boxed_val() as *mut HeapBin) };
            let old_bin_ptr = bin.bytes();
            let old_bin_base = self.buffer.base;
            let base_offset = distance_absolute(old_bin_ptr, old_bin_base);
            let layout = Layout::new::<Self>();
            unsafe {
                // Allocate space for header
                let ptr = process.alloc_layout(layout).unwrap().as_ptr() as *mut Self;
                // Write header, with modifications
                let mut buffer = self.buffer;
                buffer.orig = new_bin_ref.as_term();
                let new_bin_base = new_bin_ref.bytes().offset(base_offset as isize);
                buffer.base = new_bin_base;
                ptr::write(
                    ptr,
                    Self {
                        header: self.header,
                        buffer,
                        save_offset: self.save_offset,
                    },
                );
                let mc = &*ptr;
                mc.as_term()
            }
        }
    }
}
impl fmt::Debug for MatchContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MatchContext")
            .field("header", &self.header.as_usize())
            .field("buffer", &self.buffer)
            .field("save_offset", &self.save_offset)
            .finish()
    }
}

/// This function is intended for internal use only, specifically for use
/// by the garbage collector, which occasionally needs to update pointers
/// which reference the underlying bytes of a heap-allocated binary
#[inline]
pub(crate) fn binary_bytes(term: Term) -> *mut u8 {
    // This function is only intended to be called on boxed binary terms
    assert!(term.is_boxed());
    let ptr = term.boxed_val();
    let boxed = unsafe { *ptr };
    if boxed.is_heapbin() {
        let heapbin = unsafe { &*(ptr as *mut HeapBin) };
        return heapbin.bytes();
    }
    // This function is only valid if called on a procbin or a heapbin
    assert!(boxed.is_procbin());
    let procbin = unsafe { &*(ptr as *mut ProcBin) };
    procbin.bytes()
}

/// Creates a mask which can be used to extract bits from a byte
///
/// # Example
///
/// ```rust,ignore
/// let mask = make_bitmask(3);
/// assert_eq!(0b00000111, mask);
/// ```
#[inline(always)]
fn make_bitmask(n: u8) -> u8 {
    debug_assert!(n < 8);
    (1 << n) - 1
}

/// Assigns the bits from `src` to `dst` using the given mask,
/// preserving the bits in `dst` outside of the mask
///
/// # Example
///
/// ```rust,ignore
/// let mask = make_bitmask(3);
/// let src = 0b00000101;
/// let dst = 0b01000010;
/// let result = mask_bits(src, dst, mask);
/// assert_eq!(0b01000101);
/// ```
#[inline(always)]
fn mask_bits(src: u8, dst: u8, mask: u8) -> u8 {
    (src & mask) | (dst & !mask)
}

/// Returns true if the given bit in `byte` is set
///
/// # Example
///
/// ```rust,ignore
/// let byte = 0b01000000;
/// assert_eq!(is_bit_set(byte, 7), true);
/// ```
#[allow(unused)]
#[inline(always)]
fn is_bit_set(byte: u8, bit: u8) -> bool {
    byte & ((1 << (bit - 1)) >> (bit - 1)) == 1
}

/// Returns the value stored in the bit of `byte` at `offset`
#[allow(unused)]
#[inline(always)]
fn get_bit(byte: u8, offset: usize) -> u8 {
    byte >> (7 - (offset as u8)) & 1
}

/// Returns the number of bytes needed to store `bits` bits
#[inline]
fn num_bytes(bits: usize) -> usize {
    (bits + 7) >> 3
}

#[inline]
fn bit_offset(offset: usize) -> usize {
    offset & 7
}

#[inline]
fn byte_offset(offset: usize) -> usize {
    offset >> 3
}

/// Higher-level bit copy operation
///
/// This function copies `bits` bits from `src` to `dst`. If the source and destination
/// are both binaries (i.e. bit offsets are 0, and the number of bits is divisible by 8),
/// then the copy is performed using a more efficient primitive (essentially memcpy). In
/// all other cases, the copy is delegated to `copy_bits`, which handles bitstrings.
#[inline]
pub unsafe fn copy_binary_to_buffer(
    src: *mut u8,
    src_offs: usize,
    dst: *mut u8,
    dst_offs: usize,
    bits: usize,
) {
    if bit_offset(dst_offs) == 0 && src_offs == 0 && bit_offset(bits) == 0 && bits != 0 {
        let dst = dst.offset(byte_offset(dst_offs) as isize);
        ptr::copy_nonoverlapping(src, dst, num_bytes(bits));
    } else {
        copy_bits(
            src,
            src_offs,
            CopyDirection::Forward,
            dst,
            dst_offs,
            CopyDirection::Forward,
            bits,
        );
    }
}

/// This enum defines which direction to copy from/to in `copy_bits`
///
/// When copying from `Forward` to `Backward`, or vice versa,
/// the bits are reversed during the copy. When the values match,
/// then it is just a normal copy.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CopyDirection {
    Forward,
    Backward,
}
impl CopyDirection {
    #[inline]
    fn as_isize(&self) -> isize {
        match self {
            &CopyDirection::Forward => 1,
            &CopyDirection::Backward => -1,
        }
    }
}

/// Fundamental bit copy operation.
///
/// This function copies `bits` bits from `src` to `dst`. By specifying
/// the copy directions, it is possible to reverse the copied bits, see
/// the `CopyDirection` enum for more info.
pub unsafe fn copy_bits(
    src: *mut u8,
    src_offs: usize,
    src_d: CopyDirection,
    dst: *mut u8,
    dst_offs: usize,
    dst_d: CopyDirection,
    bits: usize,
) {
    if bits == 0 {
        return;
    }

    let src_di = src_d.as_isize();
    let dst_di = dst_d.as_isize();
    let mut src = src.offset(src_di * byte_offset(src_offs) as isize);
    let mut dst = dst.offset(dst_di * byte_offset(dst_offs) as isize);
    let src_offs = bit_offset(src_offs);
    let dst_offs = bit_offset(dst_offs);
    let dste_offs = bit_offset(dst_offs + bits);
    let lmask = if dst_offs > 0 {
        make_bitmask(8 - dst_offs as u8)
    } else {
        0
    };
    let rmask = if dste_offs > 0 {
        make_bitmask(dst_offs as u8) << (8 - dst_offs) as u8
    } else {
        0
    };

    // Take care of the case that all bits are in the same byte
    if dst_offs + bits < 8 {
        let lmask = if (lmask & rmask) > 0 {
            lmask & rmask
        } else {
            lmask | rmask
        };

        if src_offs == dst_offs {
            ptr::write(dst, mask_bits(*src, *dst, lmask));
        } else if src_offs > dst_offs {
            let mut n = *src << (src_offs - dst_offs);
            if src_offs + bits > 8 {
                src = src.offset(src_di);
                n |= *src >> (8 - (src_offs - dst_offs));
            }
            ptr::write(dst, mask_bits(n, *dst, lmask));
        } else {
            ptr::write(dst, mask_bits(*src >> (dst_offs - src_offs), *dst, lmask));
        }

        return;
    }

    // Beyond this point, we know that the bits span at least 2 bytes or more
    let mut count = (if lmask > 0 {
        bits - (8 - dst_offs)
    } else {
        bits
    }) >> 3;
    if src_offs == dst_offs {
        // The bits are aligned in the same way. We can just copy the bytes,
        // except the first and last.
        //
        // NOTE: The directions might be different, so we can't use `ptr::copy`

        if lmask > 0 {
            ptr::write(dst, mask_bits(*src, *dst, lmask));
            dst = dst.offset(dst_di);
            src = src.offset(src_di);
        }

        while count > 0 {
            count -= 1;
            ptr::write(dst, *src);
            dst = dst.offset(dst_di);
            src = src.offset(src_di);
        }

        if rmask > 0 {
            ptr::write(dst, mask_bits(*src, *dst, rmask));
        }
    } else {
        // The tricky case - the bits must be shifted into position
        let lshift;
        let rshift;
        let mut src_bits;
        let mut src_bits1;

        if src_offs > dst_offs {
            lshift = src_offs - dst_offs;
            rshift = 8 - lshift;
            src_bits = *src;
            if src_offs + bits > 8 {
                src = src.offset(src_di);
            }
        } else {
            rshift = dst_offs - src_offs;
            lshift = 8 - rshift;
            src_bits = 0;
        }

        if lmask > 0 {
            src_bits1 = src_bits << lshift;
            src_bits = *src;
            src = src.offset(src_di);
            src_bits1 |= src_bits >> rshift;
            ptr::write(dst, mask_bits(src_bits1, *dst, lmask));
            dst = dst.offset(dst_di);
        }

        while count > 0 {
            count -= 1;
            src_bits1 = src_bits << lshift;
            src_bits = *src;
            src = src.offset(src_di);
            ptr::write(dst, src_bits1 | (src_bits >> rshift));
            dst = dst.offset(dst_di);
        }

        if rmask > 0 {
            src_bits1 = src_bits << lshift;
            if ((rmask << rshift) & 0xff) > 0 {
                src_bits = *src;
                src_bits1 |= src_bits >> rshift;
            }
            ptr::write(dst, mask_bits(src_bits1, *dst, rmask));
        }
    }
}

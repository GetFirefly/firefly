use core::alloc::Layout;
use core::cmp;
use core::mem;
use core::ptr::{self, NonNull};
use core::slice;
use core::str;
use core::sync::atomic;
use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(test))]
use intrusive_collections::LinkedListLink;

use super::follow_moved;
use super::{AsTerm, Term};

pub trait Binary {
    fn as_bytes(&self) -> &[u8];
}

/// This is the header written alongside all procbin binaries in the heap,
/// it owns the refcount and has the pointer to the data and its size
#[repr(C)]
pub struct ProcBinInner {
    refc: AtomicUsize,
    size: usize,
    data: *mut u8,
}
impl ProcBinInner {
    #[inline]
    pub fn data(&self) -> *mut u8 {
        self.data
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

const FLAG_IS_RAW_BIN: usize = 0;
const FLAG_IS_LATIN1_BIN: usize = 1;
const FLAG_IS_UTF8_BIN: usize = 2;
const FLAG_MASK: usize = FLAG_IS_RAW_BIN | FLAG_IS_LATIN1_BIN | FLAG_IS_UTF8_BIN;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BinaryType {
    Raw,
    Latin1,
    Utf8,
}
impl BinaryType {
    pub fn to_procbin_header(&self) -> usize {
        match self {
            &BinaryType::Raw => Term::FLAG_PROCBIN | FLAG_IS_RAW_BIN,
            &BinaryType::Latin1 => Term::FLAG_PROCBIN | FLAG_IS_LATIN1_BIN,
            &BinaryType::Utf8 => Term::FLAG_PROCBIN | FLAG_IS_LATIN1_BIN | FLAG_IS_UTF8_BIN,
        }
    }

    pub fn to_heapbin_header(&self) -> usize {
        match self {
            &BinaryType::Raw => Term::FLAG_HEAPBIN | FLAG_IS_RAW_BIN,
            &BinaryType::Latin1 => Term::FLAG_HEAPBIN | FLAG_IS_LATIN1_BIN,
            &BinaryType::Utf8 => Term::FLAG_HEAPBIN | FLAG_IS_LATIN1_BIN | FLAG_IS_UTF8_BIN,
        }
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
    header: usize,
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

    /// Like `from_raw`, but treats this operation as a cast, rather than
    /// as the creation of a new reference. In other words, the reference count
    /// for the binary is not incremented, hence the name.
    ///
    /// Because this is effectively a clone operation, the intrusive link stored
    /// in the original value is cloned as a fresh "unlinked" link, so it is
    /// disconnected from any intrusive lists the original was in.
    ///
    /// NOTE: This is intended for use solely by the memory management subsystem,
    /// normal usage should always acquire a reference via `from_raw` or `clone`.
    pub(crate) unsafe fn from_raw_noincrement(term: *mut ProcBin) -> Self {
        let pb = &*term;
        Self {
            header: pb.header,
            inner: pb.inner,
            link: LinkedListLink::new(),
        }
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
    pub(crate) unsafe fn into_raw(self) -> *mut ProcBin {
        let ptr = &self as *const _ as *mut _;
        mem::forget(self);
        ptr
    }

    /// Returns true if this binary is a raw binary
    #[inline]
    pub fn is_raw(&self) -> bool {
        self.header & FLAG_MASK == 0
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.header & FLAG_IS_LATIN1_BIN == FLAG_IS_LATIN1_BIN
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    pub fn is_utf8(&self) -> bool {
        self.header & FLAG_IS_UTF8_BIN == FLAG_IS_UTF8_BIN
    }

    /// Returns a `BinaryType` representing the encoding type of this binary
    #[inline]
    pub fn binary_type(&self) -> BinaryType {
        if self.is_utf8() {
            BinaryType::Utf8
        } else if self.is_latin1() {
            BinaryType::Latin1
        } else {
            BinaryType::Raw
        }
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
    pub(crate) fn data(&self) -> *mut u8 {
        self.inner().data()
    }

    /// Creates a new procbin from a str slice, by copying it to the heap
    pub fn from_str(s: &str) -> Self {
        use liblumen_core::sys::alloc as sys_alloc;

        let size = s.len();
        let (layout, _) = Layout::new::<ProcBinInner>()
            .extend(unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<u8>()) })
            .unwrap();
        let ptr = unsafe { sys_alloc::alloc(layout).unwrap().as_ptr() };
        let header_ptr = ptr as *mut ProcBinInner;
        unsafe {
            // For efficient checks on binary type later, store flags in the pointer
            let data_ptr = ptr.offset(1) as *mut u8;
            let flags = if s.is_ascii() {
                FLAG_IS_LATIN1_BIN | FLAG_IS_UTF8_BIN
            } else {
                FLAG_IS_UTF8_BIN
            };
            ptr::write(
                ptr as *mut ProcBinInner,
                ProcBinInner {
                    refc: AtomicUsize::new(1),
                    size,
                    data: data_ptr,
                },
            );
            ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, size);

            Self {
                header: flags | Term::FLAG_PROCBIN,
                inner: NonNull::new_unchecked(header_ptr),
                link: LinkedListLink::new(),
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
            let inner = self.inner();
            let bytes = slice::from_raw_parts(inner.data(), inner.size());
            str::from_utf8_unchecked(bytes)
        }
    }

    #[inline]
    fn inner(&self) -> &ProcBinInner {
        unsafe { self.inner.as_ref() }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        use liblumen_core::sys::alloc as sys_alloc;

        // Destroy the data at this time, even though we may not free the box
        // allocation itself (there may still be weak pointers lying around).

        if self.inner().refc.fetch_sub(1, Ordering::Release) == 1 {
            atomic::fence(Ordering::Acquire);
            let data = self.inner().data();
            let size = self.inner().size();
            sys_alloc::free(
                data,
                Layout::from_size_align_unchecked(size, mem::align_of::<usize>()),
            );
        }
    }
}
impl Clone for ProcBin {
    #[inline]
    fn clone(&self) -> Self {
        self.inner().refc.fetch_add(1, Ordering::AcqRel);

        ProcBin {
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
            slice::from_raw_parts(inner.data(), inner.size())
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
        Term::from_raw((self as *const _ as usize) | Term::FLAG_BOXED)
    }
}

/// Process heap allocated binary, smaller than 64 bytes
#[derive(Debug, Clone)]
#[repr(C)]
pub struct HeapBin {
    header: usize,
    size: usize,
    ptr: *mut u8,
}

impl HeapBin {
    /// Returns true if this binary is a raw binary
    #[inline]
    pub fn is_raw(&self) -> bool {
        self.header & FLAG_MASK == 0
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.header & FLAG_IS_LATIN1_BIN == FLAG_IS_LATIN1_BIN
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    pub fn is_utf8(&self) -> bool {
        self.header & FLAG_IS_UTF8_BIN == FLAG_IS_UTF8_BIN
    }

    /// Returns a `BinaryType` representing the encoding type of this binary
    #[inline]
    pub fn binary_type(&self) -> BinaryType {
        if self.is_utf8() {
            BinaryType::Utf8
        } else if self.is_latin1() {
            BinaryType::Latin1
        } else {
            BinaryType::Raw
        }
    }

    /// Returns the size of this binary in bytes
    #[inline]
    pub fn size(&self) -> usize {
        self.size
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
    pub(crate) fn data(&self) -> *mut u8 {
        self.ptr
    }

    /// Reifies a `HeapBin` from a raw, untagged, pointer
    #[inline]
    pub unsafe fn from_raw(term: *mut HeapBin) -> Self {
        let hb = &*term;
        hb.clone()
    }

    /// Creates a new `HeapBin` from a raw pointer and size, which
    /// points to a valid Latin-1 binary.
    ///
    /// # Safety
    ///
    /// This function is unsafe because the `HeapBin` returned will be
    /// treated as safe to use in functions which expect latin-1 or utf-8
    /// encoded strings. You _must_ ensure that this invariant is guaranteed
    /// by the caller.
    #[inline]
    pub unsafe fn from_raw_latin1_parts(ptr: *mut u8, size: usize) -> Self {
        let flags = FLAG_IS_LATIN1_BIN | Term::FLAG_HEAPBIN;
        Self::from_raw_heapbin_parts(flags, ptr, size)
    }

    /// Creates a new `HeapBin` from a raw pointer and size, which
    /// points to a valid UTF-8 binary.
    ///
    /// # Safety
    ///
    /// This function is unsafe because the `HeapBin` returned will be
    /// treated as safe to use in functions which expect latin-1 or utf-8
    /// encoded strings. You _must_ ensure that this invariant is guaranteed
    /// by the caller.
    #[inline]
    pub unsafe fn from_raw_utf8_parts(ptr: *mut u8, size: usize) -> Self {
        let flags = FLAG_IS_UTF8_BIN | Term::FLAG_HEAPBIN;
        Self::from_raw_heapbin_parts(flags, ptr, size)
    }

    /// Creates a `HeapBin` from the raw parts of another `HeapBin`
    ///
    /// # Safety
    ///
    /// This function is unsafe for all kinds of reasons, it is intended
    /// solely for use by the memory management subsystem, use the other
    /// `from_raw_*` APIs to create `HeapBin`s from raw binary data
    #[inline]
    pub unsafe fn from_raw_heapbin_parts(header: usize, ptr: *mut u8, size: usize) -> Self {
        assert!(header & Term::FLAG_HEAPBIN == Term::FLAG_HEAPBIN);
        Self { header, size, ptr }
    }

    /// Converts this binary to a `&str` slice.
    ///
    /// This conversion does not move the string, it can be considered as
    /// creating a new reference with a lifetime attached to that of `self`.
    #[allow(unused)]
    #[inline]
    pub fn as_str<'a>(&'a self) -> &'a str {
        assert!(
            self.is_latin1() || self.is_utf8(),
            "cannot convert a binary containing non-UTF-8/non-ASCII characters to &str"
        );
        unsafe {
            let bytes = slice::from_raw_parts(self.ptr, self.size);
            str::from_utf8_unchecked(bytes)
        }
    }
}

impl Binary for HeapBin {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
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
        Term::from_raw((self as *const _ as usize) | Term::FLAG_BOXED)
    }
}

/// A slice of a binary
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SubBinary {
    header: usize,
    // Binary size in bytes
    size: usize,
    // Offset into original binary
    offset: usize,
    // Size of binary in bits
    bitsize: u8,
    // Offset in bits
    bitoffs: u8,
    // Indicates the underlying binary is writable
    writable: bool,
    // Original binary term (ProcBin or HeapBin)
    orig: Term,
}
impl SubBinary {
    pub unsafe fn from_raw(ptr: *mut SubBinary) -> Self {
        *ptr
    }

    /// During garbage collection, we sometimes want to convert sub-binary terms
    /// into full-fledged heap binaries, so that the original full-size binary can be freed.
    ///
    /// If this sub-binary is a candidate for conversion, then it will return `Ok((ptr, size))`,
    /// otherwise it will return `Err(())`. The returned pointer and size is sufficient for
    /// passing to `ptr::copy_non_overlapping` during creation of the new HeapBin.
    ///
    /// NOTE: You should not use this for any other purpose
    pub(crate) fn to_heapbin_parts(&self) -> Result<(usize, *mut u8, usize), ()> {
        if self.bitsize == 0
            && self.bitoffs == 0
            && !self.writable
            && self.size <= mem::size_of::<Term>() * 3
        {
            Ok(unsafe { self.to_raw_parts() })
        } else {
            Err(())
        }
    }

    #[inline]
    unsafe fn to_raw_parts(&self) -> (usize, *mut u8, usize) {
        let real_bin_ptr = follow_moved(self.orig).boxed_val();
        let real_bin = *real_bin_ptr;
        if real_bin.is_procbin() {
            let bin = &*(real_bin_ptr as *mut ProcBin);
            let bytes = bin.data().offset(self.offset as isize);
            let header = bin.binary_type().to_heapbin_header();
            (header, bytes, self.size)
        } else {
            assert!(real_bin.is_heapbin());
            let bin = &*(real_bin_ptr as *mut HeapBin);
            let bytes = bin.data().offset(self.offset as isize);
            let header = bin.binary_type().to_heapbin_header();
            (header, bytes, self.size)
        }
    }
}

unsafe impl AsTerm for SubBinary {
    unsafe fn as_term(&self) -> Term {
        Term::from_raw((self as *const _ as usize) | Term::FLAG_SUBBINARY)
    }
}

impl Binary for SubBinary {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            let (_header, ptr, size) = self.to_raw_parts();
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

/// Used in match contexts
///
/// This is a combination of the BinMatchState and BinMatchBuffer structs
/// from OTP, but has essentially the same set of fields, albeit with different
/// names
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MatchContext {
    header: usize,
    // Original binary term
    orig: Term,
    // Current position in binary
    base: *mut u8,
    // Size of binary in bits
    bitsize: usize,
    // Offset in bits
    bitoffset: usize,
    // Saved offsets for contexts created via `bs_start_match2`
    save_offset: Option<Term>,
}
impl MatchContext {
    #[inline]
    pub unsafe fn from_raw(ptr: *mut MatchContext) -> Self {
        *ptr
    }

    /// Used by garbage collection to get a pointer to the original
    /// term in order to place/modify move markers
    #[inline]
    pub(crate) fn orig(&self) -> *mut Term {
        &self.orig as *const _ as *mut Term
    }

    /// Used by garbage collection to get a pointer to the raw binary
    /// data pointer in order to update it if the underlying binary moves
    #[inline]
    pub(crate) fn base(&self) -> *mut *mut u8 {
        &self.base as *const _ as *mut *mut u8
    }

    #[inline]
    unsafe fn to_raw_parts(&self) -> (usize, *mut u8, usize) {
        let real_bin_ptr = follow_moved(self.orig).boxed_val();
        let real_bin = *real_bin_ptr;
        if real_bin.is_procbin() {
            let bin = &*(real_bin_ptr as *mut ProcBin);
            let bytes = bin.data().offset(byte_offset(self.bitoffset) as isize);
            let header = bin.binary_type().to_heapbin_header();
            (header, bytes, num_bytes(self.bitsize))
        } else {
            assert!(real_bin.is_heapbin());
            let bin = &*(real_bin_ptr as *mut HeapBin);
            let bytes = bin.data().offset(byte_offset(self.bitoffset) as isize);
            let header = bin.binary_type().to_heapbin_header();
            (header, bytes, num_bytes(self.bitsize))
        }
    }
}

unsafe impl AsTerm for MatchContext {
    unsafe fn as_term(&self) -> Term {
        Term::from_raw((self as *const _ as usize) | Term::FLAG_MATCH_CTX)
    }
}

impl Binary for MatchContext {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            let (_header, ptr, size) = self.to_raw_parts();
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
        return heapbin.data();
    }
    // This function is only valid if called on a procbin or a heapbin
    assert!(boxed.is_procbin());
    let procbin = unsafe { &*(ptr as *mut ProcBin) };
    procbin.data()
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

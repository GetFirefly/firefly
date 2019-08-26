use core::alloc::Layout;
use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::mem;
use core::ptr;
use core::slice;
use core::str;

use alloc::borrow::ToOwned;
use alloc::string::String;

use crate::borrow::CloneToProcess;
use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
use crate::erts::term::term::Term;
use crate::erts::term::{to_word_size, AsTerm, Boxed, TypeError, TypedTerm};
use crate::erts::string::Encoding;
use crate::erts::HeapAlloc;

use super::aligned_binary::AlignedBinary;
use super::constants::*;
use super::{ProcBin, BinaryLiteral};
use super::{Bitstring, Original};


/// Process heap allocated binary, smaller than 64 bytes
#[derive(Debug, Clone)]
#[repr(C)]
pub struct HeapBin {
    pub(super) header: Term,
    flags: usize,
}

impl HeapBin {
    pub const MAX_SIZE: usize = 64;
    // The size of the extra fields in bytes
    const EXTRA_BYTE_LEN: usize = mem::size_of::<Self>() - mem::size_of::<usize>();

    /// Create a new `HeapBin` header which will point to a binary of size `size`
    #[inline]
    pub fn new(size: usize) -> Self {
        let words = to_word_size(size) + to_word_size(Self::EXTRA_BYTE_LEN);
        Self {
            header: Term::make_header(words, Term::FLAG_HEAPBIN),
            flags: size | FLAG_IS_RAW_BIN,
        }
    }

    /// Like `new`, but for latin1-encoded binaries
    #[inline]
    pub fn new_latin1(size: usize) -> Self {
        let words = to_word_size(size) + to_word_size(Self::EXTRA_BYTE_LEN);
        Self {
            header: Term::make_header(words, Term::FLAG_HEAPBIN),
            flags: size | FLAG_IS_LATIN1_BIN,
        }
    }
    /// Like `new`, but for utf8-encoded binaries
    #[inline]
    pub fn new_utf8(size: usize) -> Self {
        let words = to_word_size(size) + to_word_size(Self::EXTRA_BYTE_LEN);
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
        self.flags & BIN_TYPE_MASK == FLAG_IS_RAW_BIN
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.flags & BIN_TYPE_MASK == FLAG_IS_LATIN1_BIN
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    pub fn is_utf8(&self) -> bool {
        self.flags & BIN_TYPE_MASK == FLAG_IS_UTF8_BIN
    }

    /// Returns a `Encoding` representing the encoding type of this binary
    #[inline]
    pub fn encoding(&self) -> Encoding {
        super::encoding_from_flags(self.flags)
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
        unsafe { (self as *const Self).add(1) as *mut u8 }
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
            let bytes = slice::from_raw_parts(self.bytes(), self.full_byte_len());
            str::from_utf8_unchecked(bytes)
        }
    }
}

impl Bitstring for HeapBin {
    fn full_byte_len(&self) -> usize {
        self.flags & !FLAG_MASK
    }
}

impl AlignedBinary for HeapBin {
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.bytes(), self.full_byte_len()) }
    }
}

impl AlignedBinary for Boxed<HeapBin> {
    fn as_bytes(&self) -> &[u8] {
        self.as_ref().as_bytes()
    }
}

impl Original for HeapBin {
    fn byte(&self, index: usize) -> u8 {
        let full_byte_len = self.full_byte_len();

        assert!(
            index < full_byte_len,
            "index ({}) >= full_byte_len ({})",
            index,
            full_byte_len
        );

        unsafe { *self.bytes().add(index) }
    }
}

unsafe impl AsTerm for HeapBin {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl CloneToProcess for HeapBin {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        let bin_size = self.full_byte_len();
        let words = self.size_in_words();
        unsafe {
            // Allocate space for header + binary
            let ptr = heap.alloc(words)?.as_ptr() as *mut Self;
            // Copy header
            ptr::copy_nonoverlapping(self as *const Self, ptr, mem::size_of::<Self>());
            // Copy binary
            let bin_ptr = ptr.add(1) as *mut u8;
            ptr::copy_nonoverlapping(self.bytes(), bin_ptr, bin_size);
            // Return term
            Ok(Term::make_boxed(ptr))
        }
    }

    fn size_in_words(&self) -> usize {
        to_word_size(self.size() + mem::size_of::<Self>())
    }
}

impl Eq for HeapBin {}

impl PartialEq<ProcBin> for Boxed<HeapBin> {
    fn eq(&self, other: &ProcBin) -> bool {
        other.eq(self)
    }
}

impl PartialEq<BinaryLiteral> for Boxed<HeapBin> {
    fn eq(&self, other: &BinaryLiteral) -> bool {
        other.eq(self)
    }
}

impl PartialOrd<ProcBin> for Boxed<HeapBin> {
    fn partial_cmp(&self, other: &ProcBin) -> Option<cmp::Ordering> {
        other.partial_cmp(self).map(|ordering| ordering.reverse())
    }
}

impl PartialOrd<BinaryLiteral> for Boxed<HeapBin> {
    fn partial_cmp(&self, other: &BinaryLiteral) -> Option<cmp::Ordering> {
        other.partial_cmp(self).map(|ordering| ordering.reverse())
    }
}

impl TryFrom<Term> for Boxed<HeapBin> {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Boxed<HeapBin> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Boxed(boxed) => boxed.to_typed_term().unwrap().try_into(),
            TypedTerm::HeapBinary(heap_binary) => Ok(heap_binary),
            _ => Err(TypeError),
        }
    }
}

impl TryInto<String> for Boxed<HeapBin> {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        match str::from_utf8(self.as_bytes()) {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => Err(badarg!()),
        }
    }
}

impl TryInto<Vec<u8>> for Boxed<HeapBin> {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<Vec<u8>, Self::Error> {
        Ok(self.as_bytes().to_vec())
    }
}

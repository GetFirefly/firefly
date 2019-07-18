use core::alloc::{AllocErr, Layout};
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
use crate::erts::term::term::Term;
use crate::erts::term::{to_word_size, AsTerm, TypeError, TypedTerm};
use crate::erts::HeapAlloc;

use super::{
    BinaryType, Bitstring, Original, FLAG_IS_LATIN1_BIN, FLAG_IS_RAW_BIN, FLAG_IS_UTF8_BIN,
    FLAG_MASK,
};

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
            let bytes = slice::from_raw_parts(self.bytes(), self.full_byte_len());
            str::from_utf8_unchecked(bytes)
        }
    }
}

impl Bitstring for HeapBin {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.bytes(), self.full_byte_len()) }
    }

    fn full_byte_len(&self) -> usize {
        (self.header.arityval() * mem::size_of::<usize>()) - Self::EXTRA_BYTE_LEN
    }

    fn partial_byte_bit_len(&self) -> u8 {
        0
    }

    fn total_bit_len(&self) -> usize {
        self.full_byte_len() * 8
    }

    fn total_byte_len(&self) -> usize {
        self.full_byte_len()
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

        unsafe { *self.bytes().offset(index as isize) }
    }
}

impl<B: Bitstring> PartialEq<B> for HeapBin {
    #[inline]
    fn eq(&self, other: &B) -> bool {
        self.as_bytes().eq(other.as_bytes())
    }
}
impl Eq for HeapBin {}
impl<B: Bitstring> PartialOrd<B> for HeapBin {
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
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, AllocErr> {
        let bin_size = self.full_byte_len();
        let words = self.size_in_words();
        unsafe {
            // Allocate space for header + binary
            let ptr = heap.alloc(words)?.as_ptr() as *mut Self;
            // Copy header
            ptr::copy_nonoverlapping(self as *const Self, ptr, mem::size_of::<Self>());
            // Copy binary
            let bin_ptr = ptr.offset(1) as *mut u8;
            ptr::copy_nonoverlapping(self.bytes(), bin_ptr, bin_size);
            // Return term
            Ok(Term::make_boxed(ptr))
        }
    }

    fn size_in_words(&self) -> usize {
        to_word_size(self.size() + mem::size_of::<Self>())
    }
}

impl TryFrom<Term> for HeapBin {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for HeapBin {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::HeapBinary(heap_binary) => Ok(heap_binary),
            _ => Err(TypeError),
        }
    }
}

impl TryInto<String> for HeapBin {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        match str::from_utf8(self.as_bytes()) {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => Err(badarg!()),
        }
    }
}

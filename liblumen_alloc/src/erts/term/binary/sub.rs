use core::alloc::Layout;
use core::convert::TryFrom;
use core::fmt;
use core::ptr;
use core::slice;

use alloc::boxed::Box;

use crate::borrow::CloneToProcess;
use crate::erts;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::TermAlloc;
use crate::erts::string::Encoding;
use crate::erts::term::prelude::*;

use super::prelude::{bit_offset, byte_offset};

/// A slice of a binary
#[derive(Clone, Copy)]
#[repr(C)]
pub struct SubBinary {
    header: Header<SubBinary>,
    /// Byte offset into `original` binary
    byte_offset: usize,
    /// Offset in bits after the `byte_offset`
    bit_offset: u8,
    /// Number of full bytes in binary.  Does not include final byte to store `bit_size` bits.
    full_byte_len: usize,
    /// Number of bits in the partial byte after the `full_byte_len` full bytes.
    partial_byte_bit_len: u8,
    /// Indicates the underlying binary is writable
    writable: bool,
    /// Original binary term (`ProcBin`, `BinaryLiteral` or `HeapBin`)
    original: Term,
}
impl_static_header!(SubBinary, Term::HEADER_SUBBINARY);
impl fmt::Debug for SubBinary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SubBinary")
            .field("header", &self.header)
            .field("byte_offset", &self.byte_offset)
            .field("bit_offset", &self.bit_offset)
            .field("full_byte_len", &self.full_byte_len)
            .field("partial_byte_bit_len", &self.partial_byte_bit_len)
            .field("writable", &self.writable)
            .field("is_binary", &self.is_binary())
            .field("is_aligned", &self.is_aligned())
            .field("original", &self.original)
            .finish()
    }
}
impl SubBinary {
    /// See erts_bs_get_binary_2 in erl_bits.c:460
    #[inline]
    pub fn from_match(ctx: &mut MatchContext, bit_len: usize) -> Self {
        assert!(ctx.buffer.bit_len - ctx.buffer.bit_offset < bit_len);

        let original = ctx.buffer.original;
        let subbinary_byte_offset = byte_offset(ctx.buffer.bit_offset);
        let subbinary_bit_offset = bit_offset(ctx.buffer.bit_offset) as u8;
        let full_byte_len = byte_offset(bit_len);
        let partial_byte_bit_len = bit_offset(bit_len) as u8;
        let writable = false;
        ctx.buffer.bit_offset += bit_len as usize;

        Self {
            header: Default::default(),
            original,
            byte_offset: subbinary_byte_offset,
            bit_offset: subbinary_bit_offset,
            full_byte_len,
            partial_byte_bit_len,
            writable,
        }
    }

    #[inline]
    pub fn from_original(
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        full_byte_len: usize,
        partial_byte_bit_len: u8,
    ) -> Self {
        Self {
            header: Default::default(),
            original,
            byte_offset,
            bit_offset,
            full_byte_len,
            partial_byte_bit_len,
            writable: false,
        }
    }

    #[inline]
    pub fn bit_offset(&self) -> u8 {
        self.bit_offset
    }

    #[inline]
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    #[inline]
    pub fn original(&self) -> Term {
        self.original
    }

    /// During garbage collection, we sometimes want to convert sub-binary terms
    /// into full-fledged heap binaries, so that the original full-size binary can be freed.
    ///
    /// If this sub-binary is a candidate for conversion, then it will return `Ok((ptr, size))`,
    /// otherwise it will return `Err(())`. The returned pointer and size is sufficient for
    /// passing to `ptr::copy_nonoverlapping` during creation of the new HeapBin.
    ///
    /// NOTE: You should not use this for any other purpose
    pub(in crate::erts) fn to_heapbin_parts(&self) -> Result<(BinaryFlags, *mut u8, usize), ()> {
        if self.is_binary()
            && self.is_aligned()
            && !self.writable
            && self.full_byte_len <= HeapBin::MAX_SIZE
        {
            Ok(unsafe { self.to_raw_parts() })
        } else {
            Err(())
        }
    }

    #[inline]
    unsafe fn to_raw_parts(&self) -> (BinaryFlags, *mut u8, usize) {
        let len = self.full_byte_len;
        match self.original.follow_moved().decode().unwrap() {
            TypedTerm::ProcBin(bin) => {
                let bytes = bin.as_byte_ptr().add(self.byte_offset);
                let flags = BinaryFlags::new(Encoding::Raw).set_size(len);
                (flags, bytes, len)
            }
            TypedTerm::BinaryLiteral(bin) => {
                let bytes = bin.as_byte_ptr().add(self.byte_offset);
                let flags = BinaryFlags::new_literal(Encoding::Raw).set_size(len);
                (flags, bytes, len)
            }
            TypedTerm::HeapBinary(bin) => {
                let bytes = bin.as_byte_ptr().add(self.byte_offset);
                let flags = BinaryFlags::new(Encoding::Raw).set_size(len);
                (flags, bytes, len)
            }
            t => panic!("invalid term, expected binary but got {:?}", t),
        }
    }

    /// Iterator of the [bit_size] bits.  To get the [byte_size] bytes at the beginning of the
    /// bitstring use [byte_iter] if the subbinary may not have [bit_offset] `0` or [as_bytes] has
    /// [bit_offset] `0`.
    pub fn partial_byte_bit_iter(&self) -> Box<dyn BitIterator> {
        let current_byte_offset = self.byte_offset + self.full_byte_len;
        let current_bit_offset = self.bit_offset;

        let improper_bit_offset = current_bit_offset + self.partial_byte_bit_len;
        let max_byte_offset = current_byte_offset + (improper_bit_offset / 8) as usize;
        let max_bit_offset = improper_bit_offset % 8;

        match self.original.decode().unwrap() {
            TypedTerm::ProcBin(bin_ptr) => Box::new(PartialByteBitIter::new(
                bin_ptr,
                current_byte_offset,
                current_bit_offset,
                max_byte_offset,
                max_bit_offset,
            )),
            TypedTerm::BinaryLiteral(bin_ptr) => Box::new(PartialByteBitIter::new(
                bin_ptr,
                current_byte_offset,
                current_bit_offset,
                max_byte_offset,
                max_bit_offset,
            )),
            TypedTerm::HeapBinary(bin_ptr) => Box::new(PartialByteBitIter::new(
                bin_ptr,
                current_byte_offset,
                current_bit_offset,
                max_byte_offset,
                max_bit_offset,
            )),
            t => panic!("invalid term, expected binary but got {:?}", t),
        }
    }

    /// Iterator for the [byte_len] bytes.  For the [partial_byte_bit_len] bits in the partial byte
    /// at the end, use [partial_byte_bit_iter].
    pub fn full_byte_iter(&self) -> Box<dyn ByteIterator<'static>> {
        let bit_offset = self.bit_offset;
        let base_byte_offset = self.byte_offset;
        let current_byte_offset = 0;
        let max_byte_offset = self.full_byte_len;

        match self.original.decode().unwrap() {
            TypedTerm::ProcBin(bin_ptr) => Box::new(FullByteIter::new(
                bin_ptr,
                base_byte_offset,
                bit_offset,
                current_byte_offset,
                max_byte_offset,
            )),
            TypedTerm::BinaryLiteral(bin_ptr) => Box::new(FullByteIter::new(
                bin_ptr,
                base_byte_offset,
                bit_offset,
                current_byte_offset,
                max_byte_offset,
            )),
            TypedTerm::HeapBinary(bin_ptr) => Box::new(FullByteIter::new(
                bin_ptr,
                base_byte_offset,
                bit_offset,
                current_byte_offset,
                max_byte_offset,
            )),
            t => panic!("invalid term, expected binary but got {:?}", t),
        }
    }
}

impl Bitstring for SubBinary {
    #[inline]
    fn full_byte_len(&self) -> usize {
        self.full_byte_len
    }

    #[inline]
    unsafe fn as_byte_ptr(&self) -> *mut u8 {
        match self.original.decode().unwrap() {
            TypedTerm::ProcBin(bin_ptr) => bin_ptr.as_ref().as_byte_ptr(),
            TypedTerm::BinaryLiteral(bin_ptr) => bin_ptr.as_ref().as_byte_ptr(),
            TypedTerm::HeapBinary(bin_ptr) => bin_ptr.as_ref().as_byte_ptr(),
            t => panic!("invalid term, expected binary but got {:?}", t),
        }
    }
}

impl CloneToProcess for SubBinary {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        let layout = Layout::new::<Self>();
        let size = layout.size();
        match self.original.follow_moved().decode().unwrap() {
            // For ref-counted binaries and those that are already on the process heap,
            // we just need to copy the sub binary header, not the binary as well
            TypedTerm::ProcBin(_bin) => {
                // Allocate space for header and copy it
                unsafe {
                    let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
                    ptr::copy_nonoverlapping(self as *const Self, ptr, size);
                    Ok(ptr.into())
                }
            }
            TypedTerm::HeapBinary(bin) => {
                if heap.is_owner(bin.as_ptr()) {
                    // Allocate space for header and copy it
                    unsafe {
                        let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
                        ptr::copy_nonoverlapping(self as *const Self, ptr, size);
                        Ok(ptr.into())
                    }
                } else {
                    // Need to make sure that the heapbin is cloned as well, and that the header is
                    // suitably updated
                    let new_bin = bin.clone_to_heap(heap)?;
                    unsafe {
                        // Allocate space for header
                        let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
                        // Write header, with modifications
                        ptr::write(
                            ptr,
                            Self {
                                header: self.header,
                                original: new_bin,
                                byte_offset: self.byte_offset,
                                bit_offset: self.bit_offset,
                                full_byte_len: self.full_byte_len,
                                partial_byte_bit_len: self.partial_byte_bit_len,
                                writable: self.writable,
                            },
                        );

                        Ok(ptr.into())
                    }
                }
            }
            t => panic!("expected ProcBin or HeapBin, but got {:?}", t),
        }
    }

    fn size_in_words(&self) -> usize {
        // Worst-case size if original also needs to be cloned
        erts::to_word_size(Layout::for_value(self).size()) + self.original.size_in_words()
    }
}

impl Eq for SubBinary {}

impl MaybeAlignedMaybeBinary for SubBinary {
    /// This is only safe to call if `is_aligned` and `is_binary`.
    ///
    /// If `is_binary`, but not not `is_aligned` use `byte_iter`.
    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        let (_flags, ptr, size) = self.to_raw_parts();

        slice::from_raw_parts(ptr, size.into())
    }

    fn is_aligned(&self) -> bool {
        self.bit_offset == 0
    }

    fn is_binary(&self) -> bool {
        self.partial_byte_bit_len == 0
    }
}

impl MaybePartialByte for SubBinary {
    #[inline]
    fn partial_byte_bit_len(&self) -> u8 {
        self.partial_byte_bit_len
    }

    #[inline]
    fn total_bit_len(&self) -> usize {
        self.full_byte_len * 8 + (self.partial_byte_bit_len as usize)
    }

    fn total_byte_len(&self) -> usize {
        self.full_byte_len + if self.is_binary() { 0 } else { 1 }
    }
}

impl MaybePartialByte for Boxed<SubBinary> {
    #[inline(always)]
    fn partial_byte_bit_len(&self) -> u8 {
        self.as_ref().partial_byte_bit_len()
    }
    #[inline(always)]
    fn total_bit_len(&self) -> usize {
        self.as_ref().total_bit_len()
    }
    #[inline(always)]
    fn total_byte_len(&self) -> usize {
        self.as_ref().total_byte_len()
    }
}

impl PartialOrd<Boxed<MatchContext>> for SubBinary {
    #[inline]
    fn partial_cmp(&self, other: &Boxed<MatchContext>) -> Option<core::cmp::Ordering> {
        self.partial_cmp(other.as_ref())
    }
}
impl PartialOrd<MatchContext> for SubBinary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn partial_cmp(&self, other: &MatchContext) -> Option<core::cmp::Ordering> {
        if self.is_binary() && other.is_binary() {
            if self.is_aligned() && other.is_aligned() {
                unsafe {
                    self.as_bytes_unchecked()
                        .partial_cmp(other.as_bytes_unchecked())
                }
            } else {
                self.full_byte_iter().partial_cmp(other.full_byte_iter())
            }
        } else {
            let bytes_partial_ordering = if self.is_aligned() && other.is_aligned() {
                unsafe {
                    self.as_bytes_unchecked()
                        .partial_cmp(other.as_bytes_unchecked())
                }
            } else {
                self.full_byte_iter().partial_cmp(other.full_byte_iter())
            };

            match bytes_partial_ordering {
                Some(core::cmp::Ordering::Equal) => self
                    .partial_byte_bit_iter()
                    .partial_cmp(other.partial_byte_bit_iter()),
                _ => bytes_partial_ordering,
            }
        }
    }
}

impl TryFrom<TypedTerm> for Boxed<SubBinary> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::SubBinary(subbinary) => Ok(subbinary),
            _ => Err(TypeError),
        }
    }
}

use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug};
use core::iter::FusedIterator;
use core::str;

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::vec::Vec;

use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
use crate::erts::term::binary::maybe_aligned_maybe_binary::MaybeAlignedMaybeBinary;

use super::*;

pub struct PartialByteBitIter {
    original: Term,
    current_byte_offset: usize,
    current_bit_offset: u8,
    max_byte_offset: usize,
    max_bit_offset: u8,
}

impl Iterator for PartialByteBitIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if (self.current_byte_offset == self.max_byte_offset)
            & (self.current_bit_offset == self.max_bit_offset)
        {
            None
        } else {
            let byte = match self.original.to_typed_term().unwrap() {
                TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                    TypedTerm::ProcBin(proc_bin) => proc_bin.byte(self.current_byte_offset),
                    TypedTerm::BinaryLiteral(literal_bin) => literal_bin.byte(self.current_byte_offset),
                    TypedTerm::HeapBinary(heap_binary) => {
                        heap_binary.byte(self.current_byte_offset)
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let bit = (byte >> (7 - self.current_bit_offset)) & 0b1;

            if self.current_bit_offset == 7 {
                self.current_bit_offset = 0;
                self.current_byte_offset += 1;
            } else {
                self.current_bit_offset += 1;
            }

            Some(bit)
        }
    }
}

pub struct FullByteIter {
    original: Term,
    base_byte_offset: usize,
    bit_offset: u8,
    current_byte_offset: usize,
    max_byte_offset: usize,
}

impl FullByteIter {
    fn is_aligned(&self) -> bool {
        self.bit_offset == 0
    }

    fn byte(&self, index: usize) -> u8 {
        let first_index = self.base_byte_offset + index;

        match self.original.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::ProcBin(proc_bin) => {
                    let first_byte = proc_bin.byte(first_index);

                    if self.is_aligned() {
                        first_byte
                    } else {
                        let second_byte = proc_bin.byte(first_index + 1);

                        (first_byte << self.bit_offset) | (second_byte >> (8 - self.bit_offset))
                    }
                }
                TypedTerm::BinaryLiteral(literal_bin) => {
                    let first_byte = literal_bin.byte(first_index);

                    if self.is_aligned() {
                        first_byte
                    } else {
                        let second_byte = literal_bin.byte(first_index + 1);

                        (first_byte << self.bit_offset) | (second_byte >> (8 - self.bit_offset))
                    }
                }
                TypedTerm::HeapBinary(heap_binary) => {
                    let first_byte = heap_binary.byte(first_index);

                    if 0 < self.bit_offset {
                        let second_byte = heap_binary.byte(first_index + 1);

                        (first_byte << self.bit_offset) | (second_byte >> (8 - self.bit_offset))
                    } else {
                        first_byte
                    }
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }
}

impl<'a> ByteIterator<'a> for FullByteIter {}

impl DoubleEndedIterator for FullByteIter {
    fn next_back(&mut self) -> Option<u8> {
        if self.current_byte_offset == self.max_byte_offset {
            None
        } else {
            self.max_byte_offset -= 1;
            let byte = self.byte(self.max_byte_offset);

            Some(byte)
        }
    }
}

impl ExactSizeIterator for FullByteIter {}

impl FusedIterator for FullByteIter {}

impl Iterator for FullByteIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.current_byte_offset == self.max_byte_offset {
            None
        } else {
            let byte = self.byte(self.current_byte_offset);
            self.current_byte_offset += 1;

            Some(byte)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.max_byte_offset - self.current_byte_offset;

        (size, Some(size))
    }
}

pub trait Original {
    fn byte(&self, index: usize) -> u8;
}

/// A slice of a binary
#[derive(Clone, Copy)]
#[repr(C)]
pub struct SubBinary {
    header: Term,
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
            header: Self::header(),
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
            header: Self::header(),
            original,
            byte_offset,
            bit_offset,
            full_byte_len,
            partial_byte_bit_len,
            writable: false,
        }
    }

    #[inline]
    pub unsafe fn from_raw(ptr: *mut SubBinary) -> Self {
        *ptr
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

    #[inline]
    pub fn bytes(&self) -> *mut u8 {
        let real_bin_ptr = follow_moved(self.original).boxed_val();
        let real_bin = unsafe { *real_bin_ptr };
        if real_bin.is_procbin() {
            let bin = unsafe { &*(real_bin_ptr as *mut ProcBin) };
            bin.bytes()
        } else if real_bin.is_binary_literal() {
            let bin = unsafe { &*(real_bin_ptr as *mut BinaryLiteral) };
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
    unsafe fn to_raw_parts(&self) -> (Term, usize, *mut u8, usize) {
        let real_bin_ptr = follow_moved(self.original).boxed_val();
        let real_bin = *real_bin_ptr;
        if real_bin.is_procbin() {
            let bin = &*(real_bin_ptr as *mut ProcBin);
            let bytes = bin.bytes().add(self.byte_offset);
            (bin.header, FLAG_IS_RAW_BIN, bytes, self.full_byte_len)
        } else if real_bin.is_binary_literal() {
            let bin = &*(real_bin_ptr as *mut BinaryLiteral);
            let bytes = bin.bytes().add(self.byte_offset);
            (bin.header, FLAG_IS_RAW_BIN | FLAG_IS_LITERAL, bytes, self.full_byte_len)
        } else {
            assert!(real_bin.is_heapbin());
            let bin = &*(real_bin_ptr as *mut HeapBin);
            let bytes = bin.bytes().add(self.byte_offset);
            (bin.header, FLAG_IS_RAW_BIN, bytes, self.full_byte_len)
        }
    }

    #[inline]
    fn header() -> Term {
        Term::make_header(arity_of::<Self>(), Term::FLAG_SUBBINARY)
    }
}

unsafe impl AsTerm for SubBinary {
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl Bitstring for SubBinary {
    #[inline]
    fn full_byte_len(&self) -> usize {
        self.full_byte_len
    }
}

impl CloneToProcess for SubBinary {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        let real_bin_ptr = follow_moved(self.original).boxed_val();
        let real_bin = unsafe { *real_bin_ptr };
        // For ref-counted binaries and those that are already on the process heap,
        // we just need to copy the sub binary header, not the binary as well
        if real_bin.is_procbin() || (real_bin.is_heapbin() && heap.is_owner(real_bin_ptr)) {
            let size = mem::size_of::<Self>();
            unsafe {
                // Allocate space for header and copy it
                let ptr = heap.alloc(to_word_size(size))?.as_ptr() as *mut Self;
                ptr::copy_nonoverlapping(self as *const Self, ptr, size);
                Ok(Term::make_boxed(ptr))
            }
        } else {
            assert!(real_bin.is_heapbin());
            // Need to make sure that the heapbin is cloned as well, and that the header is suitably
            // updated
            let bin = unsafe { &*(real_bin_ptr as *mut HeapBin) };
            let new_bin = bin.clone_to_heap(heap)?;
            let size = mem::size_of::<Self>();
            unsafe {
                // Allocate space for header
                let ptr = heap.alloc(to_word_size(size))?.as_ptr() as *mut Self;
                // Write header, with modifications
                ptr::write(
                    ptr,
                    Self {
                        header: self.header,
                        original: new_bin.into(),
                        byte_offset: self.byte_offset,
                        bit_offset: self.bit_offset,
                        full_byte_len: self.full_byte_len,
                        partial_byte_bit_len: self.partial_byte_bit_len,
                        writable: self.writable,
                    },
                );

                Ok(Term::make_boxed(ptr))
            }
        }
    }

    fn size_in_words(&self) -> usize {
        // Worst-case size if original also needs to be cloned
        to_word_size(mem::size_of_val(self)) + self.original.size_in_words()
    }
}

impl Debug for SubBinary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SubBinary")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("original", &self.original)
            .field("full_byte_len", &self.full_byte_len)
            .field("byte_offset", &self.byte_offset)
            .field("partial_byte_bit_len", &self.partial_byte_bit_len)
            .field("bit_offset", &self.bit_offset)
            .field("writable", &self.writable)
            .finish()
    }
}

impl Eq for SubBinary {}

impl IterableBitstring<'_, FullByteIter> for SubBinary {
    /// Iterator for the [byte_len] bytes.  For the [partial_byte_bit_len] bits in the partial byte
    /// at the end, use [partial_byte_bit_iter].
    fn full_byte_iter(&self) -> FullByteIter {
        FullByteIter {
            original: self.original,
            base_byte_offset: self.byte_offset,
            bit_offset: self.bit_offset,
            current_byte_offset: 0,
            max_byte_offset: self.full_byte_len,
        }
    }
}
impl MaybeAlignedMaybeBinary for SubBinary {
    type Iter = PartialByteBitIter;

    /// This is only safe to call if `is_aligned` and `is_binary`.
    ///
    /// If `is_binary`, but not not `is_aligned` use `byte_iter`.
    #[inline]
    unsafe fn as_bytes(&self) -> &[u8] {
        let (_header, _flags, ptr, size) = self.to_raw_parts();

        slice::from_raw_parts(ptr, size.into())
    }

    fn is_aligned(&self) -> bool {
        self.bit_offset == 0
    }

    fn is_binary(&self) -> bool {
        self.partial_byte_bit_len == 0
    }

    /// Iterator of the [bit_size] bits.  To get the [byte_size] bytes at the beginning of the
    /// bitstring use [byte_iter] if the subbinary may not have [bit_offset] `0` or [as_bytes] has
    /// [bit_offset] `0`.
    fn partial_byte_bit_iter(&self) -> PartialByteBitIter {
        let current_byte_offset = self.byte_offset + self.full_byte_len;
        let current_bit_offset = self.bit_offset;

        let improper_bit_offset = current_bit_offset + self.partial_byte_bit_len;
        let max_byte_offset = current_byte_offset + (improper_bit_offset / 8) as usize;
        let max_bit_offset = improper_bit_offset % 8;

        PartialByteBitIter {
            original: self.original,
            current_byte_offset,
            current_bit_offset,
            max_byte_offset,
            max_bit_offset,
        }
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

impl PartialOrd<MatchContext> for SubBinary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn partial_cmp(&self, other: &MatchContext) -> Option<core::cmp::Ordering> {
        if self.is_binary() && other.is_binary() {
            if self.is_aligned() && other.is_aligned() {
                unsafe { self.as_bytes().partial_cmp(other.as_bytes()) }
            } else {
                self.full_byte_iter().partial_cmp(other.full_byte_iter())
            }
        } else {
            let bytes_partial_ordering = if self.is_aligned() && other.is_aligned() {
                unsafe { self.as_bytes().partial_cmp(other.as_bytes()) }
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

impl TryFrom<Term> for SubBinary {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for SubBinary {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Boxed(boxed) => boxed.to_typed_term().unwrap().try_into(),
            TypedTerm::SubBinary(subbinary) => Ok(subbinary),
            _ => Err(TypeError),
        }
    }
}

impl TryInto<String> for SubBinary {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        if self.is_binary() {
            if self.is_aligned() {
                let bytes = unsafe { self.as_bytes() };

                match str::from_utf8(bytes) {
                    Ok(s) => Ok(s.to_owned()),
                    Err(_) => Err(badarg!()),
                }
            } else {
                let byte_vec: Vec<u8> = self.full_byte_iter().collect();

                String::from_utf8(byte_vec).map_err(|_| badarg!())
            }
        } else {
            Err(badarg!())
        }
    }
}

impl TryInto<Vec<u8>> for SubBinary {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<Vec<u8>, Self::Error> {
        if self.is_binary() {
            if self.is_aligned() {
                Ok(unsafe { self.as_bytes() }.to_vec())
            } else {
                Ok(self.full_byte_iter().collect())
            }
        } else {
            Err(badarg!())
        }
    }
}

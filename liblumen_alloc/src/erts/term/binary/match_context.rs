use core::alloc::Layout;
use core::convert::TryInto;
use core::ptr;
use core::slice;
use core::str;

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::boxed::Box;

use liblumen_core::util::pointer::distance_absolute;

use crate::borrow::CloneToProcess;
use crate::erts::HeapAlloc;
use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
use crate::erts::term::prelude::{Term, TypedTerm, Cast, Header, Encoded};

use super::prelude::*;

/// Represents a binary being matched
///
/// See `ErlBinMatchBuffer` in `erl_bits.h`
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MatchBuffer {
    // Original binary term
    pub(super) original: Term,
    // Current position in binary
    base: *mut u8,
    // Offset in bits
    pub(super) bit_offset: usize,
    // Size of binary in bits
    pub(super) bit_len: usize,
}
impl MatchBuffer {
    /// Create a match buffer from a binary
    ///
    /// See `erts_bs_start_match_2` in `erl_bits.c`
    #[inline]
    pub fn start_match(original: Term) -> Self {
        assert!(original.is_boxed());

        let (base, full_byte_bit_len, byte_offset, bit_offset, partial_byte_bit_len) =
            match original.decode().unwrap() {
                TypedTerm::ProcBin(bin_ptr) => {
                    let bin = bin_ptr.as_ref();
                    let ptr = unsafe { bin.as_byte_ptr() };
                    (ptr, bin.full_byte_len() * 8, 0, 0, 0)
                }
                TypedTerm::BinaryLiteral(bin_ptr) => {
                    let bin = bin_ptr.as_ref();
                    let ptr = unsafe { bin.as_byte_ptr() };
                    (ptr, bin.full_byte_len() * 8, 0, 0, 0)
                }
                TypedTerm::HeapBinary(bin_ptr) => {
                    let bin = bin_ptr.as_ref();
                    let ptr = unsafe { bin.as_byte_ptr() };
                    (ptr, bin.full_byte_len() * 8, 0, 0, 0)
                }
                TypedTerm::SubBinary(bin_ptr) => {
                    let bin = bin_ptr.as_ref();
                    let ptr = unsafe { bin.as_byte_ptr() };
                    (
                        ptr,
                        bin.full_byte_len() * 8,
                        bin.byte_offset(),
                        bin.bit_offset(),
                        bin.partial_byte_bit_len(),
                    )
                }
                t => panic!("expected valid binary term, but got {:?}", t)
            };

        let improper_bit_offset = 8 * byte_offset + (bit_offset as usize);
        let bit_len = full_byte_bit_len + improper_bit_offset + (partial_byte_bit_len as usize);
        Self {
            original,
            base,
            bit_offset: improper_bit_offset,
            bit_len,
        }
    }
}

/// Used in match contexts
///
/// See `ErlBinMatchState` and `ErlBinMatchBuffer` in `erl_bits.h`
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MatchContext {
    header: Header<MatchContext>,
    pub(super) buffer: MatchBuffer,
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

        let save_offset = if buffer.bit_offset > 0 {
            Some(buffer.bit_offset)
        } else {
            None
        };

        Self {
            header: Default::default(),
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
        &self.buffer.original as *const _ as *mut Term
    }

    /// Used by garbage collection to get a pointer to the raw binary
    /// data pointer in order to update it if the underlying binary moves
    #[inline]
    pub(crate) fn base(&self) -> *mut *mut u8 {
        &self.buffer.base as *const _ as *mut *mut u8
    }

    #[inline]
    unsafe fn to_raw_parts(&self) -> (BinaryFlags, *mut u8, usize) {
        let size = num_bytes(self.buffer.bit_len);
        match self.buffer.original.decode().unwrap() {
            TypedTerm::ProcBin(bin) => {
                let bytes = bin.as_byte_ptr().add(byte_offset(self.buffer.bit_offset));
                let flags = BinaryFlags::new(bin.encoding())
                    .set_size(size);
                (flags, bytes, size)
            }
            TypedTerm::BinaryLiteral(bin) => {
                let bytes = bin.as_byte_ptr().add(byte_offset(self.buffer.bit_offset));
                let flags = BinaryFlags::new_literal(bin.encoding())
                    .set_size(size);
                (flags, bytes, size)
            }
            TypedTerm::HeapBinary(bin) => {
                let bytes = bin.as_byte_ptr().add(byte_offset(self.buffer.bit_offset));
                let flags = BinaryFlags::new(bin.encoding())
                    .set_size(size);
                (flags, bytes, size)
            }
            t => panic!("invalid term, expected binary but got {:?}", t)
        }
    }

    /// Iterator of the [bit_size] bits.  To get the [byte_size] bytes at the beginning of the
    /// bitstring use [byte_iter] if the subbinary may not have [bit_offset] `0` or [as_bytes] has
    /// [bit_offset] `0`.
    pub fn partial_byte_bit_iter(&self) -> Box<dyn BitIterator> {
        let current_bit_offset = self.buffer.bit_offset;
        let current_byte_offset = byte_offset(current_bit_offset) + self.full_byte_len();

        let improper_bit_offset = (current_bit_offset as u8) + self.partial_byte_bit_len();
        let max_byte_offset = current_byte_offset + (improper_bit_offset / 8) as usize;
        let max_bit_offset = improper_bit_offset % 8;

        match self.buffer.original.decode().unwrap() {
            TypedTerm::ProcBin(bin_ptr) => {
                Box::new(PartialByteBitIter::new(
                    bin_ptr,
                    current_byte_offset,
                    current_bit_offset as u8,
                    max_byte_offset,
                    max_bit_offset
                ))
            }
            TypedTerm::BinaryLiteral(bin_ptr) => {
                Box::new(PartialByteBitIter::new(
                    bin_ptr,
                    current_byte_offset,
                    current_bit_offset as u8,
                    max_byte_offset,
                    max_bit_offset
                ))
            }
            TypedTerm::HeapBinary(bin_ptr) => {
                Box::new(PartialByteBitIter::new(
                    bin_ptr,
                    current_byte_offset,
                    current_bit_offset as u8,
                    max_byte_offset,
                    max_bit_offset
                ))
            }
            t => panic!("invalid term, expected binary but got {:?}", t),
        }
    }

    /// Iterator for the [byte_len] bytes.  For the [partial_byte_bit_len] bits in the partial byte
    /// at the end, use [partial_byte_bit_iter].
    #[inline]
    pub fn full_byte_iter(&self) -> Box<dyn ByteIterator<'static>> {
        let bit_offset = self.buffer.bit_offset as u8;
        let base_byte_offset = byte_offset(bit_offset as usize);
        let current_byte_offset = 0;
        let max_byte_offset = self.full_byte_len();

        match self.buffer.original.decode().unwrap() {
            TypedTerm::ProcBin(bin_ptr) => {
                Box::new(FullByteIter::new(
                    bin_ptr,
                    base_byte_offset,
                    bit_offset,
                    current_byte_offset,
                    max_byte_offset
                ))
            }
            TypedTerm::BinaryLiteral(bin_ptr) => {
                Box::new(FullByteIter::new(
                    bin_ptr,
                    base_byte_offset,
                    bit_offset,
                    current_byte_offset,
                    max_byte_offset
                ))
            }
            TypedTerm::HeapBinary(bin_ptr) => {
                Box::new(FullByteIter::new(
                    bin_ptr,
                    base_byte_offset,
                    bit_offset,
                    current_byte_offset,
                    max_byte_offset
                ))
            }
            t => panic!("invalid term, expected binary but got {:?}", t)
        }
    }
}

impl Bitstring for MatchContext {
    fn full_byte_len(&self) -> usize {
        self.buffer.bit_len / 8
    }

    unsafe fn as_byte_ptr(&self) -> *mut u8 {
        self.buffer.base
    }
}

impl CloneToProcess for MatchContext {
    fn clone_to_heap<A>(&self, heap: &mut A) -> Result<Term, Alloc>
    where
        A: ?Sized + HeapAlloc,
    {
        // For ref-counted binaries and those that are already on the process heap,
        // we just need to copy the match context header, not the binary as well.
        // Likewise with binary literals
        let layout = Layout::new::<Self>();
        let size = layout.size();
        match self.buffer.original.decode().unwrap() {
            TypedTerm::BinaryLiteral(_bin) => {
                unsafe {
                    // Allocate space for header and copy it
                    let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
                    ptr::copy_nonoverlapping(self as *const Self, ptr, size);
                    Ok(ptr.into())
                }
            }
            TypedTerm::ProcBin(_bin) => {
                unsafe {
                    // Allocate space for header and copy it
                    let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
                    ptr::copy_nonoverlapping(self as *const Self, ptr, size);
                    Ok(ptr.into())
                }
            }
            TypedTerm::HeapBinary(bin) => {
                if heap.is_owner(bin.as_ptr()) {
                    unsafe {
                        // Allocate space for header and copy it
                        let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
                        ptr::copy_nonoverlapping(self as *const Self, ptr, size);
                        Ok(ptr.into())
                    }
                } else {
                    // Need to make sure that the heapbin is cloned as well, and that the header is suitably
                    // updated
                    let bin_size = bin.size();
                    let new_bin = bin.clone_to_heap(heap)?;
                    let new_bin_ptr: *mut Term = new_bin.dyn_cast();
                    let new_bin_box = unsafe { HeapBin::from_raw_parts(new_bin_ptr as *mut u8, bin_size) };
                    let new_bin_ref = new_bin_box.as_ref();
                    let old_bin_ptr = unsafe { bin.as_byte_ptr() };
                    let old_bin_base = self.buffer.base;
                    let base_offset = distance_absolute(old_bin_ptr, old_bin_base);
                    unsafe {
                        // Allocate space for header
                        let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
                        // Write header, with modifications
                        let mut buffer = self.buffer;
                        buffer.original = new_bin;
                        let new_bin_base = new_bin_ref.as_byte_ptr().add(base_offset);
                        buffer.base = new_bin_base;
                        ptr::write(
                            ptr,
                            Self {
                                header: self.header,
                                buffer,
                                save_offset: self.save_offset,
                            },
                        );
                        Ok(ptr.into())
                    }
                }
            }
            t => panic!("expected binary term, but got {:?}", t)
        }
    }
}

impl Eq for MatchContext {}

impl MaybeAlignedMaybeBinary for MatchContext {
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        let (_flags, ptr, size) = self.to_raw_parts();

        slice::from_raw_parts(ptr, size)
    }

    fn is_aligned(&self) -> bool {
        self.buffer.bit_offset % 8 == 0
    }

    fn is_binary(&self) -> bool {
        self.buffer.bit_len % 8 == 0
    }
}

impl MaybePartialByte for MatchContext {
    fn partial_byte_bit_len(&self) -> u8 {
        (self.buffer.bit_len % 8) as u8
    }

    fn total_bit_len(&self) -> usize {
        self.buffer.bit_len
    }

    fn total_byte_len(&self) -> usize {
        (self.buffer.bit_len + (8 - 1)) / 8
    }
}

impl TryInto<String> for MatchContext {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        if self.is_binary() {
            if self.is_aligned() {
                match str::from_utf8(unsafe { self.as_bytes_unchecked() }) {
                    Ok(s) => Ok(s.to_owned()),
                    Err(_) => Err(badarg!()),
                }
            } else {
                String::from_utf8(self.full_byte_iter().collect()).map_err(|_| badarg!())
            }
        } else {
            Err(badarg!())
        }
    }
}

impl TryInto<String> for &MatchContext {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        (*self).try_into()
    }
}

impl TryInto<Vec<u8>> for MatchContext {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<Vec<u8>, Self::Error> {
        if self.is_binary() {
            if self.is_aligned() {
                Ok(unsafe { self.as_bytes_unchecked() }.to_owned())
            } else {
                Ok(self.full_byte_iter().collect())
            }
        } else {
            Err(badarg!())
        }
    }
}

impl TryInto<Vec<u8>> for &MatchContext {
    type Error = runtime::Exception;

    #[inline]
    fn try_into(self) -> Result<Vec<u8>, Self::Error> {
        (*self).try_into()
    }
}

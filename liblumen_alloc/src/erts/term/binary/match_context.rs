use core::alloc::AllocErr;
use core::cmp;
use core::convert::TryInto;
use core::fmt;
use core::mem;
use core::ptr;
use core::slice;
use core::str;

use alloc::borrow::ToOwned;
use alloc::string::String;

use liblumen_core::util::pointer::distance_absolute;

use crate::borrow::CloneToProcess;
use crate::erts::exception::runtime;
use crate::erts::term::term::Term;
use crate::erts::term::{follow_moved, to_word_size, AsTerm, HeapBin, ProcBin, SubBinary};
use crate::erts::HeapAlloc;

use super::{byte_offset, num_bytes, Bitstring};

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

        let bin_ptr = original.boxed_val();
        let bin = unsafe { *bin_ptr };

        let (base, full_byte_bit_len, byte_offset, bit_offset, partial_byte_bit_len) =
            if bin.is_procbin() {
                let pb = unsafe { &*(bin_ptr as *mut ProcBin) };
                (pb.bytes(), pb.full_byte_len() * 8, 0, 0, 0)
            } else if bin.is_heapbin() {
                let hb = unsafe { &*(bin_ptr as *mut HeapBin) };
                (hb.bytes(), hb.full_byte_len() * 8, 0, 0, 0)
            } else {
                assert!(bin.is_subbinary());
                let sb = unsafe { &*(bin_ptr as *mut SubBinary) };
                (
                    sb.bytes(),
                    sb.full_byte_len() * 8,
                    sb.byte_offset(),
                    sb.bit_offset(),
                    sb.partial_byte_bit_len(),
                )
            };

        let improper_bit_offset = 8 * byte_offset + (bit_offset as usize);
        Self {
            original,
            base,
            bit_offset: improper_bit_offset,
            bit_len: full_byte_bit_len + improper_bit_offset + (partial_byte_bit_len as usize),
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
        &self.buffer.original as *const _ as *mut Term
    }

    /// Used by garbage collection to get a pointer to the raw binary
    /// data pointer in order to update it if the underlying binary moves
    #[inline]
    pub(crate) fn base(&self) -> *mut *mut u8 {
        &self.buffer.base as *const _ as *mut *mut u8
    }

    #[inline]
    unsafe fn to_raw_parts(&self) -> (Term, usize, *mut u8, usize) {
        let real_bin_ptr = follow_moved(self.buffer.original).boxed_val();
        let real_bin = *real_bin_ptr;
        if real_bin.is_procbin() {
            let bin = &*(real_bin_ptr as *mut ProcBin);
            let bytes = bin
                .bytes()
                .offset(byte_offset(self.buffer.bit_offset) as isize);
            let flags = bin.binary_type().to_flags();
            (bin.header, flags, bytes, num_bytes(self.buffer.bit_len))
        } else {
            assert!(real_bin.is_heapbin());
            let bin = &*(real_bin_ptr as *mut HeapBin);
            let bytes = bin
                .bytes()
                .offset(byte_offset(self.buffer.bit_offset) as isize);
            let flags = bin.binary_type().to_flags();
            (bin.header, flags, bytes, num_bytes(self.buffer.bit_len))
        }
    }
}
unsafe impl AsTerm for MatchContext {
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl Bitstring for MatchContext {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            let (_header, _flags, ptr, size) = self.to_raw_parts();
            slice::from_raw_parts(ptr, size)
        }
    }

    fn full_byte_len(&self) -> usize {
        self.buffer.bit_len / 8
    }

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

impl<B: Bitstring> PartialEq<B> for MatchContext {
    #[inline]
    fn eq(&self, other: &B) -> bool {
        self.as_bytes().eq(other.as_bytes())
    }
}
impl Eq for MatchContext {}
impl<B: Bitstring> PartialOrd<B> for MatchContext {
    #[inline]
    fn partial_cmp(&self, other: &B) -> Option<cmp::Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

impl CloneToProcess for MatchContext {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, AllocErr> {
        let real_bin_ptr = follow_moved(self.buffer.original).boxed_val();
        let real_bin = unsafe { *real_bin_ptr };
        // For ref-counted binaries and those that are already on the process heap,
        // we just need to copy the match context header, not the binary as well
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
            let new_bin_ref = unsafe { &*(new_bin.boxed_val() as *mut HeapBin) };
            let old_bin_ptr = bin.bytes();
            let old_bin_base = self.buffer.base;
            let base_offset = distance_absolute(old_bin_ptr, old_bin_base);
            let size = mem::size_of::<Self>();
            unsafe {
                // Allocate space for header
                let ptr = heap.alloc(to_word_size(size))?.as_ptr() as *mut Self;
                // Write header, with modifications
                let mut buffer = self.buffer;
                buffer.original = new_bin_ref.as_term();
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
                Ok(Term::make_boxed(ptr))
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

impl TryInto<String> for MatchContext {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        match str::from_utf8(self.as_bytes()) {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => Err(badarg!()),
        }
    }
}

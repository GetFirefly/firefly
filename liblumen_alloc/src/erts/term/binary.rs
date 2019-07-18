mod heap;
mod match_context;
mod process;
mod sub;

use core::cmp;
use core::mem;
use core::ptr;
use core::slice;

use crate::borrow::CloneToProcess;

use super::*;

pub use heap::HeapBin;
pub use match_context::MatchContext;
pub use process::ProcBin;
pub use sub::{Original, SubBinary};

pub trait Bitstring {
    fn as_bytes(&self) -> &[u8];

    /// The total number of full bytes, not including any final partial byte.
    fn full_byte_len(&self) -> usize;

    /// The number of bits in the partial byte.
    fn partial_byte_bit_len(&self) -> u8;

    /// The total number of bits include those in bytes and any bits in a partial byte.
    fn total_bit_len(&self) -> usize;

    /// The total of number of bytes needed to hold `total_bit_len`
    fn total_byte_len(&self) -> usize;
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

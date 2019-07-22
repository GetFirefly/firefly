mod heap;
mod match_context;
mod process;
mod sub;

use core::mem;
use core::ptr;
use core::slice;

use crate::borrow::CloneToProcess;

use super::*;

pub use heap::HeapBin;
pub use match_context::MatchContext;
pub use process::ProcBin;
pub use sub::{Original, SubBinary};

struct PartialByteBitIter {
    byte: u8,
    current_bit_offset: u8,
    max_bit_offset: u8,
}

impl PartialByteBitIter {
    fn new(byte: u8, bit_len: u8) -> Self {
        Self {
            byte,
            current_bit_offset: 0,
            max_bit_offset: bit_len,
        }
    }
}

impl Iterator for PartialByteBitIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.current_bit_offset == self.max_bit_offset {
            None
        } else {
            let bit = (self.byte >> (7 - self.current_bit_offset)) & 0b1;

            self.current_bit_offset += 1;

            Some(bit)
        }
    }
}

pub trait MaybePartialByte {
    /// The number of bits in the partial byte.
    fn partial_byte_bit_len(&self) -> u8;

    /// The total number of bits include those in bytes and any bits in a partial byte.
    fn total_bit_len(&self) -> usize;

    /// The total of number of bytes needed to hold `total_bit_len`
    fn total_byte_len(&self) -> usize;
}

pub trait Bitstring {
    /// The total number of full bytes, not including any final partial byte.
    fn full_byte_len(&self) -> usize;
}

/// A `BitString` that is guaranteed to always be a binary of aligned bytes
pub trait AlignedBinary {
    fn as_bytes(&self) -> &[u8];
}

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_eq_aligned_binary_aligned_binary {
    ($o:tt for $s:tt) => {
        impl PartialEq<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn eq(&self, other: &$o) -> bool {
                self.as_bytes().eq(other.as_bytes())
            }
        }
    };
}

partial_eq_aligned_binary_aligned_binary!(HeapBin for HeapBin);
// No (ProcBin for HeapBin) as we always reverse order to save space
partial_eq_aligned_binary_aligned_binary!(HeapBin for ProcBin);
partial_eq_aligned_binary_aligned_binary!(ProcBin for ProcBin);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_ord_aligned_binary_aligned_binary {
    ($o:tt for $s:tt) => {
        impl PartialOrd<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn partial_cmp(&self, other: &$o) -> Option<core::cmp::Ordering> {
                self.as_bytes().partial_cmp(other.as_bytes())
            }
        }
    };
}

partial_ord_aligned_binary_aligned_binary!(HeapBin for HeapBin);
// No (ProcBin for HeapBin) as we always reverse order to save space
partial_ord_aligned_binary_aligned_binary!(HeapBin for ProcBin);
partial_ord_aligned_binary_aligned_binary!(ProcBin for ProcBin);

pub trait ByteIterator<'a>: ExactSizeIterator + DoubleEndedIterator + Iterator<Item = u8>
where
    Self: Sized,
{
}

impl<'a> ByteIterator<'a> for core::iter::Copied<core::slice::Iter<'a, u8>> {}

impl<A: AlignedBinary + Bitstring> MaybePartialByte for A {
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

pub trait IterableBitstring<'a, I: ByteIterator<'a>> {
    fn full_byte_iter(&'a self) -> I;
}

impl<'a, A: AlignedBinary> IterableBitstring<'a, core::iter::Copied<core::slice::Iter<'a, u8>>>
    for A
{
    fn full_byte_iter(&'a self) -> core::iter::Copied<core::slice::Iter<'a, u8>> {
        self.as_bytes().iter().copied()
    }
}

pub trait MaybeAlignedMaybeBinary {
    type Iter: Iterator<Item = u8>;

    unsafe fn as_bytes(&self) -> &[u8];

    fn is_aligned(&self) -> bool;

    fn is_binary(&self) -> bool;

    fn partial_byte_bit_iter(&self) -> Self::Iter;
}

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_eq_aligned_binary_maybe_aligned_maybe_binary {
    ($o:tt for $s:tt) => {
        impl PartialEq<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn eq(&self, other: &$o) -> bool {
                if self.is_binary() {
                    if self.is_aligned() {
                        unsafe { self.as_bytes() }.eq(other.as_bytes())
                    } else {
                        self.full_byte_iter().eq(other.full_byte_iter())
                    }
                } else {
                    false
                }
            }
        }
    };
}

partial_eq_aligned_binary_maybe_aligned_maybe_binary!(HeapBin for MatchContext);
partial_eq_aligned_binary_maybe_aligned_maybe_binary!(ProcBin for MatchContext);
partial_eq_aligned_binary_maybe_aligned_maybe_binary!(HeapBin for SubBinary);
partial_eq_aligned_binary_maybe_aligned_maybe_binary!(ProcBin for SubBinary);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_ord_aligned_binary_maybe_aligned_maybe_binary {
    ($o:tt for $s:tt) => {
        impl PartialOrd<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn partial_cmp(&self, other: &$o) -> Option<core::cmp::Ordering> {
                use core::cmp::Ordering::*;

                let mut self_full_byte_iter = self.full_byte_iter();
                let mut other_full_byte_iter = other.full_byte_iter();
                let mut partial_ordering = Some(Equal);

                while let Some(Equal) = partial_ordering {
                    match (self_full_byte_iter.next(), other_full_byte_iter.next()) {
                        (Some(self_byte), Some(other_byte)) => {
                            partial_ordering = self_byte.partial_cmp(&other_byte)
                        }
                        (None, Some(other_byte)) => {
                            let partial_byte_bit_len = self.partial_byte_bit_len();

                            partial_ordering =
                                if partial_byte_bit_len > 0 {
                                    self.partial_byte_bit_iter().partial_cmp(
                                        PartialByteBitIter::new(other_byte, partial_byte_bit_len),
                                    )
                                } else {
                                    Some(Less)
                                };

                            break;
                        }
                        (Some(_), None) => {
                            partial_ordering = Some(Greater);

                            break;
                        }
                        (None, None) => {
                            if 0 < self.partial_byte_bit_len() {
                                partial_ordering = Some(Greater);
                            }

                            break;
                        }
                    }
                }

                partial_ordering
            }
        }
    };
}

partial_ord_aligned_binary_maybe_aligned_maybe_binary!(HeapBin for MatchContext);
partial_ord_aligned_binary_maybe_aligned_maybe_binary!(ProcBin for MatchContext);
partial_ord_aligned_binary_maybe_aligned_maybe_binary!(HeapBin for SubBinary);
partial_ord_aligned_binary_maybe_aligned_maybe_binary!(ProcBin for SubBinary);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_eq_maybe_aligned_maybe_binary_maybe_aligned_maybe_binary {
    ($o:tt for $s:ty) => {
        impl PartialEq<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn eq(&self, other: &$o) -> bool {
                if self.is_binary() && other.is_binary() {
                    if self.is_aligned() && other.is_aligned() {
                        unsafe { self.as_bytes().eq(other.as_bytes()) }
                    } else {
                        self.full_byte_iter().eq(other.full_byte_iter())
                    }
                } else {
                    let bytes_equal = if self.is_aligned() && other.is_aligned() {
                        unsafe { self.as_bytes().eq(other.as_bytes()) }
                    } else {
                        self.full_byte_iter().eq(other.full_byte_iter())
                    };

                    bytes_equal || {
                        self.partial_byte_bit_iter()
                            .eq(other.partial_byte_bit_iter())
                    }
                }
            }
        }
    };
}

partial_eq_maybe_aligned_maybe_binary_maybe_aligned_maybe_binary!(SubBinary for SubBinary);
partial_eq_maybe_aligned_maybe_binary_maybe_aligned_maybe_binary!(MatchContext for SubBinary);
partial_eq_maybe_aligned_maybe_binary_maybe_aligned_maybe_binary!(MatchContext for MatchContext);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_ord_maybe_aligned_maybe_binary_maybe_aligned_maybe_binary {
    ($o:tt for $s:ty) => {
        impl PartialOrd<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn partial_cmp(&self, other: &$o) -> Option<core::cmp::Ordering> {
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
    };
}

partial_ord_maybe_aligned_maybe_binary_maybe_aligned_maybe_binary!(SubBinary for SubBinary);
partial_ord_maybe_aligned_maybe_binary_maybe_aligned_maybe_binary!(MatchContext for SubBinary);
partial_ord_maybe_aligned_maybe_binary_maybe_aligned_maybe_binary!(MatchContext for MatchContext);

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

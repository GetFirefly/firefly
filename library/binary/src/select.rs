use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Shl;
use core::str;

use crate::helpers::*;
use crate::{BitsIter, Bitstring, ByteIter};

/// Represents some number of bits that fit within a single byte
///
/// This is used in the context of bit selections which may return
/// partial bytes, this struct allows us to introspect those partial bytes.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct MaybePartialByte {
    /// The underlying byte
    pub byte: [u8; 1],
    /// The number of significant bits, counted from the most-significant bit
    pub size: u8,
}
impl fmt::Debug for MaybePartialByte {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::fmt::Write;

        let byte = self.byte();
        if !self.is_partial() {
            if let Some(c) = char::from_u32(byte as u32) {
                return f.write_char(c);
            }
        }

        f.write_str("0b")?;
        for i in 0..self.size {
            let offset = 7 - i;
            let mask = 1 << offset;
            if byte & mask == mask {
                f.write_char('1')?;
            } else {
                f.write_char('0')?;
            }
        }
        let remaining = 8 - self.size;
        for _ in 0..remaining {
            f.write_char('X')?;
        }
        Ok(())
    }
}
impl MaybePartialByte {
    #[inline]
    pub fn new(byte: u8, size: u8) -> Self {
        assert!(size <= 8);
        Self { byte: [byte], size }
    }

    #[inline(always)]
    pub const fn byte(&self) -> u8 {
        self.byte[0]
    }

    #[inline]
    pub fn is_partial(&self) -> bool {
        self.size < 8
    }
}
impl From<u8> for MaybePartialByte {
    /// We treat conversions from u8 as whole bytes, i.e. the size is always 8
    #[inline]
    fn from(byte: u8) -> Self {
        Self {
            byte: [byte],
            size: 8,
        }
    }
}
impl Shl<usize> for MaybePartialByte {
    type Output = Self;

    fn shl(self, other: usize) -> Self::Output {
        assert!(other <= 8);
        let size = self.size.saturating_sub(other as u8);
        let byte = self.byte().checked_shl(other as u32).unwrap_or(0);
        Self::new(byte, size)
    }
}
impl Shl<u8> for MaybePartialByte {
    type Output = Self;

    fn shl(self, other: u8) -> Self::Output {
        let size = self.size.saturating_sub(other);
        let byte = self.byte().checked_shl(other as u32).unwrap_or(0);
        Self::new(byte, size)
    }
}

/// This enum represents the various types of results that can be produced by selecting
/// an arbitrary number of bits from a buffer. See the variant documentation for more
/// details on what each type of result means and how it can be produced.
///
/// The purpose of this enum is three-fold:
///
/// * Represent zero-copy bit selections of any size, over any buffer, at any offset,
/// with potentially non-binary buffer sizes.
/// * Clarify code which operates on bit selections by describing what the possible
/// combinations of selection result are and what data is relevant for them
/// * Enable optimizations when optimal (e.g. when a selection is aligned and binary)
///
/// NOTE: The selected bytes may represent a smaller number of bits than requested,
/// depending on the size of the selectable region. Typically this behavior is differentiated
/// by wrapping selections in `Result`, so that an `Ok` selection is guaranteed to
/// be precise, while an `Err` selection represents the bits that matched the selection
/// up to the end of the selectable range.
///
/// NOTE: Any partial bytes produced by the selection are guaranteed to have their extra
/// bits zeroed, i.e. it is never the case that a selection will include bits that are
/// not strictly within the selected range.
#[derive(Copy, Clone)]
pub enum Selection<'a> {
    /// The selection contains no bits
    ///
    /// This might happen either because the requested size was zero, or because the
    /// underlying selectable range was empty
    Empty,
    /// The selection contains a number of bits that fit within a single byte, along
    /// with the number of significant bits counted from the most-significant bit.
    ///
    /// This happens when either the requested size is <= 8 bits, or the underlying
    /// selectable range was <= 8 bits.
    Byte(MaybePartialByte),
    /// The selection produced a number of bits evenly divisible into multiple bytes,
    /// aligned on a byte boundary
    AlignedBinary(&'a [u8]),
    /// The selection produced a number of bits evenly divisible into multiple bytes,
    /// but the bytes are not aligned on a byte boundary; as a result, the selection
    /// produces both a leading and trailing partial byte, with at least 8 more bits
    /// in the interval between them.
    Binary(MaybePartialByte, &'a [u8], MaybePartialByte),
    /// The selection produced a number of bits that are _not_ evenly divisible into
    /// multiple bytes, but the selection begins on a byte boundary. Such a selection
    /// always has a trailing partial byte.
    AlignedBitstring(&'a [u8], MaybePartialByte),
    /// The selection produced a number of bits that are _not_ evenly divisible into
    /// multiple bytes, and furthermore, the selection is not aligned to a byte boundary.
    ///
    /// In this situation, the leading byte is guaranteed to be partial, but the trailing
    /// bit may or may not be, depending on the interaction between the bit offset and the number
    /// of selected bits.
    ///
    /// For example, given a buffer of bytes where every bit is set to 1, then:
    ///
    /// * If a selection of 9 bits starts from bit offset 3, then those 9 bits
    /// require two partial bytes to cover, so you would end up with:
    ///
    ///     Selection::Bitstring(0b00011111, &[], Some(0b11110000))
    ///
    /// * If a selection of 12 bits starts from bit offset 4, then those 12 bits
    /// require only a leading partial byte:
    ///
    ///     Selection::Bitstring(0b00001111, &[0b11111111], None)
    ///
    /// * If a selection of 13 bits starts from bit offset 4, then those 13 bits
    /// require both a leading and trailing partial byte:
    ///
    ///     Selection::Bitstring(0b00001111, &[0b11111111], Some(0b10000000))
    Bitstring(MaybePartialByte, &'a [u8], Option<MaybePartialByte>),
}
impl<'a> Selection<'a> {
    /// Creates a new selection covering all of the bytes in `data`
    pub fn all(data: &'a [u8]) -> Self {
        if data.is_empty() {
            Self::Empty
        } else {
            Self::AlignedBinary(data)
        }
    }

    /// Creates a new selection covering all of the bits in the provided bitstring
    pub fn from_bitstring(data: &'a dyn Bitstring) -> Self {
        let bit_size = data.bit_size();
        if bit_size == 0 {
            return Self::Empty;
        }
        let bytes = unsafe { data.as_bytes_unchecked() };
        Self::new(bytes, 0, data.bit_offset(), None, bit_size).unwrap()
    }

    /// This function can be used to select `n` bits from `data`, taking into account both byte
    /// and bit offsets, as well as an optional bit length constraint on `data` (i.e. the buffer
    /// is 8 bytes, but we only want to treat it as if it was 7 1/2 bytes).
    ///
    /// The result of this function (whether successful or not) is the `Selection` enum, see its
    /// documentation for details on its variants.
    ///
    /// When `Err(selection)` is returned, there were insufficient bits in the buffer to fulfill the
    /// request, or the request was out of bounds. The selection contained within will hold what
    /// was able to be selected from the available bits. This can be used to implement APIs that
    /// select as many bits as are available without having to calculate in advance the exact
    /// number of bits to ask for.
    ///
    /// NOTE: While you can use this function to perform the equivalent of a simple slicing
    /// operation (e.g. `&data[offset..len]`), it is recommended that you use slicing directly
    /// when you know the bounds and are working with aligned, binary data. This function does a
    /// lot of work to validate the selection and handle various combinations of offsets and
    /// non-binary sizes, so it is more efficient to slice directly when possible.
    pub fn new(
        data: &'a [u8],
        byte_offset: usize,
        bit_offset: u8,
        bitsize: Option<usize>,
        n: usize,
    ) -> Result<Self, Self> {
        use core::cmp;

        assert!(bit_offset < 8, "invalid bit offset, must be a value < 8; use a combination of byte_offset + bit_offset for larger offsets");

        // Empty requests are an edge case, but can be handled without any other processing
        if n == 0 {
            return Ok(Self::Empty);
        }

        let len = data.len();
        let buffer_bitsize = len * 8;
        let bitsize = cmp::min(bitsize.unwrap_or(buffer_bitsize), buffer_bitsize);
        let selectable = bitsize.saturating_sub((byte_offset * 8) + bit_offset as usize);

        // Empty buffers are an edge case, but can be handled without any other processing
        if selectable == 0 {
            return Err(Self::Empty);
        }

        // The number of trailing bits in the last selectable byte is either the number
        // of bits in that byte which are selectable, or 0, indicating that the entire
        // last byte is selectable.
        let selectable_trailing_bits = (bitsize % 8) as u8;
        let trailing_bits = (n % 8) as u8;
        let bytes = n / 8;

        // The request is aligned if the first bit falls on a byte boundary
        let is_aligned = bit_offset == 0;
        // The request is binary if the number of requested bits is evenly divisible into bytes
        let is_binary = trailing_bits == 0;
        // The buffer is binary if the addressable range is evenly divisible into bytes or if the
        // range the requested size falls within is evenly divisible into bytes
        let is_buffer_binary = if is_aligned {
            selectable_trailing_bits == 0 || selectable.saturating_sub(bit_offset as usize) >= n
        } else {
            bit_offset == selectable_trailing_bits
        };

        // OPTIMIZATION: If we can handle this request with a simple slice operation, do so
        if is_aligned && is_binary {
            if n <= selectable {
                return Ok(Self::AlignedBinary(
                    &data[byte_offset..(byte_offset + bytes)],
                ));
            } else if is_buffer_binary {
                return Err(Self::AlignedBinary(&data[byte_offset..]));
            } else if selectable <= 8 {
                let selectable = selectable as u8;
                let first = unsafe { *data.get_unchecked(byte_offset) };
                let leading_bits = 8 - bit_offset;
                if selectable <= leading_bits {
                    let mask = bitmask_be(selectable);
                    let byte = (first << bit_offset) & mask;
                    return Err(Self::Byte(MaybePartialByte::new(byte, selectable)));
                }
            } else {
                let last_index = (selectable / 8) + (selectable_trailing_bits > 0) as usize;
                let byte = unsafe { *data.get_unchecked(last_index) };
                let mask = bitmask_be(selectable_trailing_bits);
                return Err(Self::AlignedBitstring(
                    &data[byte_offset..last_index],
                    MaybePartialByte::new(byte & mask, selectable_trailing_bits),
                ));
            }
        }

        // If the request doesn't fit in the addressable range, adjust the selection to return an
        // error with the maximum selection possible
        let fits = selectable >= n;
        if !fits {
            let is_binary = selectable % 8 == 0;
            // Adjust the selection to take all remaining bits
            if is_aligned {
                // If we can just take a slice from the byte offset to the end, awesome
                if is_binary {
                    return Err(Self::AlignedBinary(&data[byte_offset..]));
                } else if selectable > 8 {
                    // The selectable region forms an aligned bitstring, so we have a trailing
                    // partial byte
                    let last_index = (selectable / 8) + byte_offset;
                    let byte = unsafe { *data.get_unchecked(last_index + 1) };
                    let mask = bitmask_be(selectable_trailing_bits);
                    return Err(Self::AlignedBitstring(
                        &data[byte_offset..last_index],
                        MaybePartialByte::new(byte & mask, selectable_trailing_bits),
                    ));
                } else {
                    let mask = bitmask_be(selectable_trailing_bits);
                    let byte = unsafe { *data.get_unchecked(byte_offset) };
                    return Err(Self::Byte(MaybePartialByte::new(
                        byte & mask,
                        selectable_trailing_bits,
                    )));
                }
            } else if is_binary {
                // The selectable region forms an unaligned binary, so we have both leading and
                // trailing partial bytes

                // Handle the case where the selected bits can be packed into a single byte
                let leading_bits = 8 - bit_offset;
                if selectable == 8 {
                    let trailing_bits = selectable as u8 - leading_bits;
                    debug_assert_eq!(leading_bits + trailing_bits, 8);
                    let x = unsafe { *data.get_unchecked(byte_offset) };
                    let y = unsafe { *data.get_unchecked(byte_offset + 1) };
                    let byte = splice_bits(x, y, bit_offset);
                    return Err(Self::Byte(MaybePartialByte::new(byte, 8)));
                }

                let remaining_bits = selectable - (leading_bits as usize);
                let trailing_bits = (remaining_bits % 8) as u8;

                let first_mask = bitmask_le(leading_bits);
                let first = unsafe { *data.get_unchecked(byte_offset) };
                let last_mask = bitmask_be(trailing_bits);
                let last_index = (selectable / 8) + byte_offset;
                let last = unsafe { *data.get_unchecked(last_index + 1) };
                return Err(Self::Binary(
                    MaybePartialByte::new((first & first_mask) << bit_offset, leading_bits),
                    &data[(byte_offset + 1)..last_index],
                    MaybePartialByte::new(last & last_mask, trailing_bits),
                ));
            } else {
                // The selectable region forms an unaligned bitstring, so we have a leading partial
                // byte, and potentially a trailing partial byte

                // Handle the case where the selected bits can be packed into a single byte
                let leading_bits = 8 - bit_offset;
                let first = unsafe { *data.get_unchecked(byte_offset) };

                if selectable <= 8 {
                    if selectable <= leading_bits as usize {
                        let selectable = selectable as u8;
                        let mask = bitmask_le(selectable);
                        let byte = (first & mask) << bit_offset;
                        // The leading byte contains all of our bits
                        return Err(Self::Byte(MaybePartialByte::new(byte, selectable)));
                    }

                    // The selected bits span two bytes
                    let last = unsafe { *data.get_unchecked(byte_offset + 1) };
                    let first_mask = bitmask_le(leading_bits);
                    let last_mask = bitmask_be(selectable_trailing_bits);
                    let byte = ((first & first_mask) << bit_offset)
                        | ((last & last_mask) >> (8 - bit_offset));
                    return Err(Self::Byte(MaybePartialByte::new(byte, selectable as u8)));
                }

                // If the remaining selectable bits ends on a byte boundary, then we don't have a
                // trailing partial byte
                let first_mask = bitmask_le(leading_bits);
                let first = (first & first_mask) << bit_offset;
                if selectable_trailing_bits == 0 {
                    return Err(Self::Bitstring(
                        MaybePartialByte::new(first, leading_bits),
                        &data[(byte_offset + 1)..],
                        None,
                    ));
                } else {
                    let last_mask = bitmask_be(selectable_trailing_bits);
                    let last_index = (selectable / 8) + byte_offset;
                    let last = unsafe { *data.get_unchecked(last_index) } & last_mask;
                    return Err(Self::Bitstring(
                        MaybePartialByte::new(first, leading_bits),
                        &data[(byte_offset + 1)..last_index],
                        Some(MaybePartialByte::new(last, selectable_trailing_bits)),
                    ));
                }
            }
        }

        // At this point we've classified the request and the underlying buffer, and we know that
        // the request is non-empty and definitely fits, all we need to do is handle the
        // various flavors of selection just like we did above, except with more accuracy as
        // we can calculate precise ranges
        if is_aligned {
            // We've already handled the case where the selection is aligned and binary, so we have
            // the following cases to handle:
            //
            // * aligned, non-binary size, binary buffer size
            // * aligned, non-binary size, non-binary buffer size
            assert!(!is_binary);
            // At this point, if the buffer is non-binary, the request requires bits from the last
            // addressable byte which is a partial byte. If this were not the case,
            // `is_buffer_binary` would be true.
            //
            // As a result, it must be the case that the number of trailing bits is less than or
            // equal to the number of addressable bits in the last byte.
            assert!(is_buffer_binary || trailing_bits <= selectable_trailing_bits);
            // The last byte is partial, but we must also handle the case where the request fits in
            // a single byte
            let mask = bitmask_be(trailing_bits);
            if n < 8 {
                let byte = unsafe { *data.get_unchecked(byte_offset) };
                Ok(Self::Byte(MaybePartialByte::new(
                    byte & mask,
                    trailing_bits,
                )))
            } else {
                let last_index = byte_offset + bytes;
                let last = unsafe { *data.get_unchecked(last_index) };
                Ok(Self::AlignedBitstring(
                    &data[byte_offset..last_index],
                    MaybePartialByte::new(last & mask, trailing_bits),
                ))
            }
        } else {
            // We have the following cases to handle here:
            //
            // * unaligned, binary size, binary buffer size
            // * unaligned, binary size, non-binary buffer size
            // * unaligned, non-binary size, binary buffer size
            // * unaligned, non-binary size, non-binary buffer size

            // The first byte is always partial here
            let leading_bits = 8 - bit_offset;
            let first_mask = bitmask_le(leading_bits);
            let first = unsafe { *data.get_unchecked(byte_offset) } & first_mask;
            // If the request fits within the first partial byte, we're done
            if n <= leading_bits as usize {
                // In this branch, we may be selecting a subset of bits within the
                // byte, so this requires a more complicated mask to extract just
                // those bits
                let n = n as u8;
                let first = first << bit_offset;
                let mask = bitmask_be(n);
                return Ok(Self::Byte(MaybePartialByte::new(first & mask, n)));
            } else if n <= 8 {
                // The remaining bits fit within a single byte
                let n = n as u8;
                let m = n - leading_bits;
                let last_mask = bitmask_be(m);
                let last = unsafe { *data.get_unchecked(byte_offset + 1) };
                let last = last & last_mask;
                let byte = (first << bit_offset) | (last >> (8 - bit_offset));
                Ok(Self::Byte(MaybePartialByte::new(byte, n)))
            } else {
                let m = n - leading_bits as usize;
                let trailing_bits = (m % 8) as u8;
                let bytes = m / 8;
                if trailing_bits == 0 {
                    // There is no trailing byte
                    let last_index = byte_offset + bytes;
                    Ok(Self::Bitstring(
                        MaybePartialByte::new(first << bit_offset, leading_bits),
                        &data[(byte_offset + 1)..(last_index + 1)],
                        None,
                    ))
                } else {
                    // There is a trailing byte
                    let last_index = byte_offset + bytes + 1;
                    let last_mask = bitmask_be(trailing_bits);
                    let last = unsafe { *data.get_unchecked(last_index) };
                    let last = last & last_mask;
                    // If the request was binary, we want to differentiate that in the result
                    if is_binary {
                        Ok(Self::Binary(
                            MaybePartialByte::new(first << bit_offset, leading_bits),
                            &data[(byte_offset + 1)..last_index],
                            MaybePartialByte::new(last, trailing_bits),
                        ))
                    } else {
                        Ok(Self::Bitstring(
                            MaybePartialByte::new(first << bit_offset, leading_bits),
                            &data[(byte_offset + 1)..last_index],
                            Some(MaybePartialByte::new(last, trailing_bits)),
                        ))
                    }
                }
            }
        }
    }

    /// Returns the byte at index `n` relative to the start of the data (including bit offset)
    ///
    /// If the index is out of range, returns `None`.
    pub fn get(&self, n: usize) -> Option<u8> {
        let byte_size = self.byte_size();
        if n >= byte_size {
            return None;
        }

        match self {
            Self::Empty => None,
            Self::Byte(b) if n == 0 => Some(b.byte()),
            Self::AlignedBinary(b) => b.get(n).copied(),
            Self::AlignedBitstring(b, r) => {
                if n >= b.len() {
                    Some(r.byte())
                } else {
                    b.get(n).copied()
                }
            }
            _ => self.bytes().nth(n),
        }
    }

    /// Returns the selected bytes as a `Cow<str>`, if the data is binary and valid UTF-8.
    ///
    /// The data is borrowed if the underlying bytes can be used directly, otherwise
    /// a new `String` value is constructed.
    ///
    /// For bitstrings, this function returns `None`
    pub fn to_str(&self) -> Option<Cow<'_, str>> {
        if !self.is_binary() {
            return None;
        }
        if self.is_aligned() {
            self.as_str().map(Cow::Borrowed)
        } else {
            let cap = self.byte_size();
            let mut buf = Vec::with_capacity(cap);
            for byte in self.bytes() {
                buf.push(byte);
            }
            String::from_utf8(buf).map(Cow::Owned).ok()
        }
    }

    /// Returns the selected bytes as a slice, if the data is aligned on a byte boundary
    ///
    /// For aligned bitstrings, this function returns a slice that covers all but the last
    /// partial byte.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Empty => Some(&[]),
            Self::Byte(b) => Some(b.byte.as_slice()),
            Self::AlignedBinary(b) => Some(b),
            _ => None,
        }
    }

    /// Returns the selection as a `Cow<[u8]>`.
    ///
    /// The data is borrowed if the underlying bytes can be directly referenced, otherwise
    /// a new `Vec` is allocated of appropriate size to hold all of the data and populated.
    ///
    /// For bitstrings, the final partial byte is zero-padded.
    pub fn to_bytes(&self) -> Cow<'_, [u8]> {
        match self {
            Self::Empty => Cow::Borrowed(&[]),
            Self::Byte(b) => Cow::Borrowed(b.byte.as_slice()),
            Self::AlignedBinary(b) => Cow::Borrowed(b),
            Self::AlignedBitstring(b, r) => {
                let cap = b.len() + 1;
                let mut vec = Vec::with_capacity(cap);
                vec.extend_from_slice(b);
                vec.push(r.byte());
                Cow::Owned(vec)
            }
            _ => {
                let cap = self.byte_size();
                let mut vec = Vec::with_capacity(cap);
                for byte in self.bytes() {
                    vec.push(byte);
                }
                Cow::Owned(vec)
            }
        }
    }

    /// Much like `to_bytes`, except this function sets aside the final partial byte,
    /// if the final byte is partial. This is only useful in a few rare situations.
    pub fn to_maybe_partial_bytes(&self) -> (Cow<'_, [u8]>, Option<MaybePartialByte>) {
        match self {
            Self::Empty => (Cow::Borrowed(&[]), None),
            Self::Byte(b) if b.is_partial() => (Cow::Borrowed(&[]), Some(*b)),
            Self::Byte(b) => (Cow::Borrowed(b.byte.as_slice()), None),
            Self::AlignedBinary(b) => (Cow::Borrowed(b), None),
            Self::AlignedBitstring(b, r) => (Cow::Borrowed(b), Some(*r)),
            _ => {
                let mut iter = self.bits();
                let mut buf = Vec::with_capacity(iter.len());
                for byte in iter.by_ref() {
                    buf.push(byte);
                }
                (Cow::Owned(buf), iter.consume())
            }
        }
    }

    /// Like `to_bytes`, but fills `buffer` with the bytes.
    ///
    /// The provided buffer must be equal to or larger in size than the selection,
    /// or this function will panic.
    ///
    /// Returns the number of trailing bits in the final byte
    pub fn write_bytes_to_buffer(&self, buffer: &mut [u8]) -> u8 {
        assert!(self.byte_size() <= buffer.len());
        match self {
            Self::Empty => 0,
            Self::Byte(b) => {
                buffer[0] = b.byte();
                8 - b.size
            }
            Self::AlignedBinary(b) => {
                let buf = &mut buffer[..b.len()];
                buf.copy_from_slice(b);
                0
            }
            Self::AlignedBitstring(b, r) => {
                let len = b.len();
                let buf = &mut buffer[..len];
                buf.copy_from_slice(b);
                buf[len] = r.byte();
                8 - r.size
            }
            _ => {
                let mut iter = self.bits();
                let mut i = 0;
                while let Some(byte) = iter.next() {
                    unsafe {
                        *buffer.get_unchecked_mut(i) = byte;
                        i += 1;
                    }
                }
                if let Some(partial) = iter.consume() {
                    unsafe {
                        *buffer.get_unchecked_mut(i) = partial.byte();
                    }
                    8 - partial.size
                } else {
                    0
                }
            }
        }
    }

    /// Pops the next byte off this selection, shrinking the selection by one byte
    pub fn pop(&mut self) -> Option<u8> {
        match self {
            Self::Empty => None,
            Self::Byte(b) => {
                let byte = b.byte();
                *self = Self::Empty;
                Some(byte)
            }
            Self::AlignedBinary(ref mut bytes) => {
                if bytes.is_empty() {
                    *self = Self::Empty;
                    return None;
                }
                bytes.take_first().copied()
            }
            Self::AlignedBitstring(ref mut bytes, last) => match bytes.len() {
                0 => {
                    let byte = last.byte();
                    *self = Self::Empty;
                    Some(byte)
                }
                1 => {
                    let byte = bytes.take_first().copied();
                    *self = Self::Byte(*last);
                    byte
                }
                _ => bytes.take_first().copied(),
            },
            Self::Binary(l, mut bytes, r) => {
                let x = l.byte();
                if bytes.is_empty() {
                    let next = r.byte();
                    let y = next >> l.size;
                    *self = Self::Empty;
                    return Some(x | y);
                }
                let next = bytes.take_first().copied().unwrap();
                let y = next >> l.size;
                let byte = x | y;
                let remaining_bits = 8 - l.size;
                let next = next << remaining_bits;
                let l = MaybePartialByte::new(next, l.size);
                *self = Self::Binary(l, bytes, *r);
                Some(byte)
            }
            Self::Bitstring(l, mut bytes, None) => {
                let x = l.byte();
                if bytes.is_empty() {
                    *self = Self::Empty;
                    return Some(x);
                }
                let next = bytes.take_first().copied().unwrap();
                let y = next >> l.size;
                let byte = x | y;
                let remaining_bits = 8 - l.size;
                let next = next << remaining_bits;
                let l = MaybePartialByte::new(next, l.size);
                *self = Self::Bitstring(l, bytes, None);
                Some(byte)
            }
            Self::Bitstring(l, mut bytes, Some(r)) => {
                let x = l.byte();
                if bytes.is_empty() {
                    let next = r.byte();
                    let y = next >> l.size;
                    let byte = x | y;
                    let combined_bits = l.size + r.size;
                    if combined_bits > 8 {
                        let extra_bits = combined_bits - 8;
                        let next = next << (8 - l.size);
                        let l = MaybePartialByte::new(next, extra_bits);
                        *self = Self::Byte(l);
                        return Some(byte);
                    } else {
                        *self = Self::Empty;
                        return Some(byte);
                    }
                }

                let next = bytes.take_first().copied().unwrap();
                let y = next >> l.size;
                let byte = x | y;
                let remaining_bits = 8 - l.size;
                let next = next << remaining_bits;
                let l = MaybePartialByte::new(next, l.size);
                *self = Self::Bitstring(l, bytes, Some(*r));
                Some(byte)
            }
        }
    }

    /// Produces a new selection that represents shrinking the current selection by `n` bits,
    /// starting from the first bit.
    pub fn shrink_front(&self, n: usize) -> Self {
        let available = self.bit_size();
        if n >= available {
            return Self::Empty;
        }
        let byte_offset = n / 8;
        let bit_offset = (n % 8) as u8;
        match self {
            Self::Empty => unreachable!(),
            Self::Byte(b) => Self::Byte((*b) << n),
            Self::AlignedBinary(bytes) => {
                match Self::new(bytes, byte_offset, bit_offset, None, available) {
                    Ok(s) => s,
                    Err(s) => s,
                }
            }
            Self::AlignedBitstring(bytes, r) => {
                let leading_byte_len = bytes.len();
                let leading_bit_len = leading_byte_len * 8;
                if n == leading_bit_len {
                    // We're consuming all the leading bits
                    Self::Byte(*r)
                } else if n < leading_bit_len {
                    // We can shrink the leading bytes and preserve everything else
                    Self::AlignedBitstring(&bytes[(leading_byte_len - 1)..], *r)
                } else {
                    // We're consuming all the leading bytes plus some of the trailing bits
                    let consumed_trailing_bits = (n - leading_bit_len) as u8;
                    let remaining_bits = r.size - consumed_trailing_bits;
                    let byte = r.byte() << consumed_trailing_bits;
                    Self::Byte(MaybePartialByte::new(byte, remaining_bits))
                }
            }
            Self::Binary(l, bytes, r) => {
                if n == l.size as usize {
                    Self::AlignedBitstring(bytes, *r)
                } else if n < l.size as usize {
                    let n = n as u8;
                    Self::Binary(*l << n, bytes, *r)
                } else {
                    let n = n - l.size as usize;
                    let leading_byte_len = bytes.len();
                    let leading_bit_len = leading_byte_len * 8;
                    if n == leading_bit_len {
                        Self::Byte(*r)
                    } else if n < leading_bit_len {
                        let byte_offset = n / 8;
                        let offset = (n % 8) as u8;
                        let leading_bits = 8 - offset;
                        let first = unsafe { *bytes.get_unchecked(byte_offset) };
                        let mask = bitmask_le(leading_bits);
                        let first = (first & mask) << offset;
                        Self::Binary(
                            MaybePartialByte::new(first, leading_bits),
                            &bytes[(byte_offset + 1)..],
                            *r,
                        )
                    } else {
                        // We're going to consume some of the trailing bits
                        let consumed_trailing_bits = (n - leading_bit_len) as u8;
                        let remaining_bits = r.size - consumed_trailing_bits;
                        let byte = r.byte() << consumed_trailing_bits;
                        Self::Byte(MaybePartialByte::new(byte, remaining_bits))
                    }
                }
            }
            Self::Bitstring(l, bytes, maybe_r) => {
                if n == l.size as usize {
                    Self::AlignedBinary(bytes)
                } else if n < l.size as usize {
                    let byte = l.byte() << n;
                    Self::Bitstring(MaybePartialByte::new(byte, l.size - n as u8), bytes, None)
                } else {
                    let n = n - l.size as usize;
                    let bytes_len = bytes.len();
                    let bytes_bitsize = bytes_len * 8;
                    if n == bytes_bitsize {
                        Self::Empty
                    } else if n < bytes_bitsize {
                        // The new start begins somewhere within `bytes`
                        // If the new `n` is evenly divisible by 8, things are simple,
                        // otherwise we need to construct a new bitstring selection
                        let byte_offset = n / 8;
                        let offset = (n % 8) as u8;
                        if offset == 0 {
                            Self::AlignedBinary(&bytes[byte_offset..])
                        } else {
                            let first = unsafe { *bytes.get_unchecked(byte_offset) };
                            let mask = bitmask_le(offset);
                            let first = (first & mask) << offset;
                            Self::Bitstring(
                                MaybePartialByte::new(first, 8 - offset),
                                &bytes[(byte_offset + 1)..],
                                None,
                            )
                        }
                    } else {
                        // The new offset starts in the trailing byte, and there must be a trailing
                        // byte here
                        let r = maybe_r.unwrap();
                        let leading_byte_len = bytes.len();
                        let leading_bit_len = leading_byte_len * 8;
                        // We're consuming all the leading bytes plus some of the trailing bits
                        let consumed_trailing_bits = (n - leading_bit_len) as u8;
                        let remaining_bits = r.size - consumed_trailing_bits;
                        let byte = r.byte() << consumed_trailing_bits;
                        Self::Byte(MaybePartialByte::new(byte, remaining_bits))
                    }
                }
            }
        }
    }

    /// Produces a new selection that represents selecting the first `n` bits of the current
    /// selection
    ///
    /// This returns `Result<Selection, Selection>`, where `Err` indicates that `n` bits were not
    /// available, but provides the best selection available from the remaining data.
    pub fn take(&self, n: usize) -> Result<Self, Self> {
        if n == 0 {
            return Ok(Self::Empty);
        }
        match self {
            Self::Empty => Err(Self::Empty),
            Self::Byte(b) => {
                let bsize = b.size as usize;
                if n == bsize {
                    Ok(Self::Byte(*b))
                } else if n > bsize {
                    Err(Self::Byte(*b))
                } else {
                    let n = n as u8;
                    let byte = b.byte() & bitmask_be(n);
                    Ok(Self::Byte(MaybePartialByte::new(byte, n)))
                }
            }
            Self::AlignedBinary(bytes) => {
                let bitsize = bytes.len() * 8;
                let trailing_bits = (n % 8) as u8;
                if n == bitsize {
                    Ok(Self::AlignedBinary(bytes))
                } else if n > bitsize {
                    Err(Self::AlignedBinary(bytes))
                } else if trailing_bits == 0 {
                    let last_index = n / 8;
                    Ok(Self::AlignedBinary(&bytes[..last_index]))
                } else if n > 8 {
                    // We're going to produce an aligned bitstring, so we will have a trailing
                    // partial byte
                    let last_index = n / 8;
                    let mask = bitmask_be(trailing_bits);
                    let last = unsafe { *bytes.get_unchecked(last_index) } & mask;
                    Ok(Self::AlignedBitstring(
                        &bytes[..last_index],
                        MaybePartialByte::new(last, trailing_bits),
                    ))
                } else {
                    // We're going to produce a single partial byte
                    let n = n as u8;
                    let mask = bitmask_be(n);
                    let byte = unsafe { *bytes.get_unchecked(0) } & mask;
                    Ok(Self::Byte(MaybePartialByte::new(byte, n)))
                }
            }
            Self::Binary(l, bytes, r) => {
                let bytes_len = bytes.len();
                let bytes_bit_len = bytes_len * 8;
                let total_bits = (l.size + r.size) as usize + bytes_bit_len;
                if n > total_bits {
                    Err(Self::Binary(*l, bytes, *r))
                } else if n == l.size as usize {
                    Ok(Self::Byte(*l))
                } else if n < l.size as usize {
                    let n = n as u8;
                    let byte = l.byte() & bitmask_be(n);
                    Ok(Self::Byte(MaybePartialByte::new(byte, n)))
                } else {
                    let n = n - l.size as usize;
                    if n == bytes_bit_len {
                        Ok(Self::Bitstring(*l, bytes, None))
                    } else if n < bytes_bit_len {
                        let trailing_bits = (n % 8) as u8;
                        let last_index = n / 8;
                        if trailing_bits == 0 {
                            Ok(Self::Bitstring(*l, &bytes[..last_index], None))
                        } else {
                            let mask = bitmask_be(trailing_bits);
                            let last = unsafe { *bytes.get_unchecked(last_index + 1) } & mask;
                            if l.size + trailing_bits == 8 {
                                Ok(Self::Binary(
                                    *l,
                                    &bytes[..last_index],
                                    MaybePartialByte::new(last, trailing_bits),
                                ))
                            } else {
                                Ok(Self::Bitstring(
                                    *l,
                                    &bytes[..last_index],
                                    Some(MaybePartialByte::new(last, trailing_bits)),
                                ))
                            }
                        }
                    } else {
                        let consumed_trailing_bits = (n - bytes_bit_len) as u8;
                        let mask = bitmask_be(consumed_trailing_bits);
                        let byte = r.byte() & mask;
                        Ok(Self::Bitstring(
                            *l,
                            bytes,
                            Some(MaybePartialByte::new(byte, consumed_trailing_bits)),
                        ))
                    }
                }
            }
            Self::AlignedBitstring(bytes, r) => {
                let bytes_len = bytes.len();
                let bytes_bit_len = bytes_len * 8;
                let total_bits = r.size as usize + bytes_bit_len;
                if n > total_bits {
                    Err(Self::AlignedBitstring(bytes, *r))
                } else if n == total_bits {
                    Ok(Self::AlignedBitstring(bytes, *r))
                } else if n == bytes_bit_len {
                    Ok(Self::AlignedBinary(bytes))
                } else if n < bytes_bit_len {
                    let trailing_bits = (n % 8) as u8;
                    let last_index = n / 8;
                    if trailing_bits == 0 {
                        Ok(Self::AlignedBinary(&bytes[..last_index]))
                    } else {
                        let mask = bitmask_be(trailing_bits);
                        let last = unsafe { *bytes.get_unchecked(last_index + 1) } & mask;
                        Ok(Self::AlignedBitstring(
                            &bytes[..last_index],
                            MaybePartialByte::new(last, trailing_bits),
                        ))
                    }
                } else {
                    // We're going to produce a single byte, containing some or all of the bits from
                    // `r`
                    let consumed_trailing_bits = (n - bytes_bit_len) as u8;
                    let mask = bitmask_be(consumed_trailing_bits);
                    let byte = r.byte() & mask;
                    Ok(Self::AlignedBitstring(
                        bytes,
                        MaybePartialByte::new(byte, consumed_trailing_bits),
                    ))
                }
            }
            Self::Bitstring(l, bytes, r) => {
                let bytes_len = bytes.len();
                let bytes_bit_len = bytes_len * 8;
                let total_bits = l.size as usize
                    + bytes_bit_len
                    + r.as_ref().map(|r| r.size as usize).unwrap_or_default();
                if n > total_bits {
                    Err(Self::Bitstring(*l, bytes, *r))
                } else if n == total_bits {
                    Ok(Self::Bitstring(*l, bytes, *r))
                } else if n == l.size as usize {
                    Ok(Self::Byte(*l))
                } else if n < l.size as usize {
                    let n = n as u8;
                    let mask = bitmask_be(n);
                    let byte = l.byte() & mask;
                    Ok(Self::Byte(MaybePartialByte::new(byte, n)))
                } else {
                    let n = n - l.size as usize;
                    if n == bytes_bit_len {
                        Ok(Self::Bitstring(*l, bytes, None))
                    } else if n < bytes_bit_len {
                        let trailing_bits = (n % 8) as u8;
                        let last_index = n / 8;
                        if trailing_bits == 0 {
                            Ok(Self::Bitstring(*l, &bytes[..last_index], None))
                        } else {
                            let mask = bitmask_be(trailing_bits);
                            let last = unsafe { *bytes.get_unchecked(last_index + 1) } & mask;
                            if l.size + trailing_bits == 8 {
                                Ok(Self::Binary(
                                    *l,
                                    &bytes[..last_index],
                                    MaybePartialByte::new(last, trailing_bits),
                                ))
                            } else {
                                Ok(Self::Bitstring(
                                    *l,
                                    &bytes[..last_index],
                                    Some(MaybePartialByte::new(last, trailing_bits)),
                                ))
                            }
                        }
                    } else {
                        // We're going to take bits from the last byte, so there must be a last byte
                        let r = (*r).unwrap();
                        let consumed_trailing_bits = (n - bytes_bit_len) as u8;
                        let mask = bitmask_be(consumed_trailing_bits);
                        let byte = r.byte() & mask;
                        Ok(Self::Bitstring(
                            *l,
                            bytes,
                            Some(MaybePartialByte::new(byte, consumed_trailing_bits)),
                        ))
                    }
                }
            }
        }
    }
}
impl<'a> Bitstring for Selection<'a> {
    fn byte_size(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Byte(_) => 1,
            Self::AlignedBinary(bytes) => bytes.len(),
            Self::Binary(_, bytes, _) => 2 + bytes.len(),
            Self::AlignedBitstring(bytes, _) => 1 + bytes.len(),
            Self::Bitstring(_, bytes, None) => 1 + bytes.len(),
            Self::Bitstring(_, bytes, Some(_)) => 2 + bytes.len(),
        }
    }

    fn bit_size(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Byte(b) => b.size as usize,
            Self::AlignedBinary(bytes) => bytes.len() * 8,
            Self::Binary(l, bytes, r) => (l.size + r.size) as usize + (bytes.len() * 8),
            Self::AlignedBitstring(bytes, r) => r.size as usize + (bytes.len() * 8),
            Self::Bitstring(l, bytes, None) => l.size as usize + (bytes.len() * 8),
            Self::Bitstring(l, bytes, Some(r)) => (l.size + r.size) as usize + (bytes.len() * 8),
        }
    }

    fn bit_offset(&self) -> u8 {
        match self {
            Self::Empty => 0,
            Self::Byte(_) => 0,
            Self::AlignedBinary(_) => 0,
            Self::Binary(b, _, _) => 8 - b.size,
            Self::AlignedBitstring(_, _) => 0,
            Self::Bitstring(l, _, _) => 8 - l.size,
        }
    }

    fn trailing_bits(&self) -> u8 {
        match self {
            Self::Empty => 0,
            Self::Byte(b) => b.size,
            Self::AlignedBinary(_) => 0,
            Self::Binary(_, _, r) => r.size,
            Self::AlignedBitstring(_, r) => r.size,
            Self::Bitstring(_, _, None) => 0,
            Self::Bitstring(_, _, Some(r)) => r.size,
        }
    }

    #[inline(always)]
    fn bytes(&self) -> ByteIter<'_> {
        ByteIter::new(*self)
    }

    #[inline(always)]
    fn bits(&self) -> BitsIter<'_> {
        BitsIter::new(*self)
    }

    /// Returns the selected bytes as a string reference, if the data is binary, aligned, and valid
    /// UTF-8.
    ///
    /// For unaligned binaries or bitstrings, returns `None`. See `to_str` for an alternative
    /// available to unaligned binaries.
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Empty => Some(""),
            Self::Byte(b) => str::from_utf8(b.byte.as_slice()).ok(),
            Self::AlignedBinary(b) => str::from_utf8(b).ok(),
            _ => None,
        }
    }

    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::Byte(b) => b.byte.as_slice(),
            Self::AlignedBinary(b) => b,
            _ => {
                panic!("it is not permitted to directly access the underlying bytes of an unaligned or non-binary selection");
            }
        }
    }
}
impl<'a> Eq for Selection<'a> {}
impl<'a, T: Bitstring> PartialEq<T> for Selection<'a> {
    fn eq(&self, other: &T) -> bool {
        // An optimization: we can say for sure that if the sizes don't match,
        // the slices don't either.
        if self.bit_size() != other.bit_size() {
            return false;
        }

        // If both slices are aligned binaries, we can compare their data directly
        if self.is_aligned() && other.is_aligned() && self.is_binary() && other.is_binary() {
            let bytes = unsafe { self.as_bytes_unchecked() };
            return bytes.eq(unsafe { other.as_bytes_unchecked() });
        }

        // Otherwise we must fall back to a byte-by-byte comparison
        self.bytes().eq(other.bytes())
    }
}
impl<'a> Ord for Selection<'a> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl<'a, T: Bitstring> PartialOrd<T> for Selection<'a> {
    // We order bitstrings lexicographically
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        // Aligned binaries can be compared using the optimal built-in slice comparisons in the
        // standard lib
        if self.is_aligned() && other.is_aligned() && self.is_binary() && other.is_binary() {
            let bytes = unsafe { self.as_bytes_unchecked() };
            return Some(bytes.cmp(unsafe { other.as_bytes_unchecked() }));
        }

        // Otherwise we must comapre byte-by-byte
        Some(self.bytes().cmp(other.bytes()))
    }
}
impl<'a> Hash for Selection<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.is_aligned() && self.is_binary() {
            return Hash::hash_slice(unsafe { self.as_bytes_unchecked() }, state);
        }

        for byte in self.bytes() {
            Hash::hash(&byte, state);
        }
    }
}
impl<'a> fmt::Debug for Selection<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("Empty"),
            Self::Byte(b) => write!(f, "Byte({:?})", b),
            Self::AlignedBinary(bytes) => match str::from_utf8(bytes) {
                Ok(s) => write!(f, "AlignedBinary({})", s),
                _ => write!(f, "AlignedBinary({:?})", bytes),
            },
            Self::Binary(l, bytes, r) => match self.to_str() {
                Some(s) => write!(f, "Binary({})", s),
                None => write!(f, "Binary({:?}, {:?}, {:?})", l, bytes, r),
            },
            Self::AlignedBitstring(bytes, r) => match str::from_utf8(bytes) {
                Ok(s) => write!(f, "AlignedBitstring({}, {:?})", s, r),
                _ => write!(f, "AlignedBitstring({:?}, {:?})", bytes, r),
            },
            Self::Bitstring(l, bytes, r) => match str::from_utf8(bytes) {
                Ok(s) => write!(f, "Bitstring({:?}, {}, {:?})", l, s, r),
                _ => write!(f, "Bitstring({:?}, {:?}, {:?})", l, bytes, r),
            },
        }
    }
}
impl<'a> fmt::Display for Selection<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::fmt::Write;

        if let Some(s) = self.to_str() {
            f.write_str("<<\"")?;
            for c in s.escape_default() {
                f.write_char(c)?;
            }
            f.write_str("\">>")
        } else {
            display_bytes(self.bytes(), f)
        }
    }
}

/// An exact selection is used for verifying that a selection as a precise representation
///
/// For our purposes, this is largely to facilitate testing, but may also be useful in contexts
/// where you want the more precise notion of equality used here.
///
/// # Rationale
///
/// Equality with `Selection` is based on normalizing the byte representation of the selected
/// data and then comparing byte-by-byte (though we ensure that partial bytes are not naively
/// compared to non-partial bytes).
///
/// The problem with this is that if we want to guarantee that a selection has a particular
/// representation, this normalization works against us, as two selections with different
/// representations can still compare equal, e.g. Selection::AlignedBinary and Selection::Binary.
///
/// This is fine in the general case, since it is the behavior we expect; but for testing we want
/// to validate that the selection behavior produces specific results, which is what we aim to
/// facilitate here.
#[derive(Debug, Copy, Clone, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ExactSelection<'a>(Selection<'a>);
impl<'a> From<Selection<'a>> for ExactSelection<'a> {
    #[inline]
    fn from(selection: Selection<'a>) -> Self {
        Self(selection)
    }
}
impl<'a> Eq for ExactSelection<'a> {}
impl<'a> PartialEq for ExactSelection<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Selection::Empty, Selection::Empty) => true,
            (Selection::Empty, _) => false,
            (Selection::Byte(l), Selection::Byte(r)) => l.eq(r),
            (Selection::Byte(_), _) => false,
            (Selection::AlignedBinary(l), Selection::AlignedBinary(r)) => l.eq(r),
            (Selection::AlignedBinary(_), _) => false,
            (Selection::Binary(ll, lb, lr), Selection::Binary(rl, rb, rr)) => {
                ll.eq(rl) && lb.eq(rb) && lr.eq(rr)
            }
            (Selection::Binary(_, _, _), _) => false,
            (Selection::AlignedBitstring(lb, lr), Selection::AlignedBitstring(rb, rr)) => {
                lb.eq(rb) && lr.eq(rr)
            }
            (Selection::AlignedBitstring(_, _), _) => false,
            (Selection::Bitstring(ll, lb, lr), Selection::Bitstring(rl, rb, rr)) => {
                ll.eq(rl) && lb.eq(rb) && lr.eq(rr)
            }
            (Selection::Bitstring(_, _, _), _) => false,
        }
    }
}
impl<'a> PartialEq<Selection<'a>> for ExactSelection<'a> {
    #[inline]
    fn eq(&self, other: &Selection<'a>) -> bool {
        self.eq(&ExactSelection(*other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 'h' = 0b01101000
    // 'e' = 0b01100101
    // 'l' = 0b01101100
    // 'o' = 0b01101111
    const HELLO: &'static [u8] = b"hello".as_slice();
    const EMPTY: &'static [u8] = &[];
    const TEN: &'static [u8] = &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

    macro_rules! assert_selection {
        ($selection:expr, $expected:expr) => {
            match ($selection, $expected) {
                (Ok(l), Ok(r)) => {
                    assert_eq!(ExactSelection(l), ExactSelection(r));
                }
                (Err(l), Err(r)) => {
                    assert_eq!(ExactSelection(l), ExactSelection(r));
                }
                (l, r) => assert_eq!(l, r),
            }
        };
    }

    #[test]
    fn selection_test_empty() {
        let selection = ExactSelection(Selection::all(EMPTY));
        assert_eq!(selection, Selection::Empty);

        // This selection is artificially constrained to 0 bits, so the request can't
        // succeed, but the fallback selection should still be empty
        let selection = Selection::new(TEN, 0, 0, Some(0), 10 * 8);
        assert_selection!(selection, Err(Selection::Empty));

        // This selection requests zero bits
        let selection = Selection::new(TEN, 0, 0, None, 0);
        assert_selection!(selection, Ok(Selection::Empty));

        // This selection requests zero bits, but also starts at the end of the buffer
        let selection = Selection::new(TEN, 10, 0, None, 0);
        assert_selection!(selection, Ok(Selection::Empty));
    }

    #[test]
    fn selection_test_byte() {
        // This selection is artifically constrained to 8 bits, so the request should produce a
        // single byte Since the offset is 1, the result is that the value 1 in the second
        // byte is shifted left and becomes the value 2
        let selection = Selection::new(TEN, 0, 1, Some(9), 8);
        assert_selection!(selection, Ok(Selection::Byte(2u8.into())));

        // This selection produces a byte which spans a byte boundary
        let buf = &[0b00001111, 0b11110000];
        let selection = Selection::new(buf, 0, 4, None, 8);
        assert_selection!(selection, Ok(Selection::Byte(u8::MAX.into())));

        // This selection is for 8 bits with 4 bytes of underlying available, so cannot possibly
        // succeed, but the fallback selection is 4 bits from the end
        let buf = &[u8::MAX, u8::MAX, u8::MAX];
        let selection = Selection::new(buf, 2, 4, None, 8);
        assert_selection!(
            selection,
            Err(Selection::Byte(MaybePartialByte::new(0b11110000, 4)))
        );

        let selection = Selection::new(buf, 0, 7, None, 6);
        assert_selection!(
            selection,
            Ok(Selection::Byte(MaybePartialByte::new(0b11111100, 6)))
        );

        // Selecting beyond the selectable range, when the available range spans across two bytes
        // produces a byte
        let selection = Selection::new(buf, 1, 7, Some(21), 24);
        assert_selection!(
            selection,
            Err(Selection::Byte(MaybePartialByte::new(0b11111100, 6)))
        );
    }

    #[test]
    fn selection_test_aligned_binary() {
        // Selecting all of a byte slice is equivalent to that slice (aligned, binary by
        // construction)
        let selection = ExactSelection(Selection::all(TEN));
        assert_eq!(selection, Selection::AlignedBinary(TEN));

        // Selecting a subset of bytes of an aligned binary is also an aligned binary
        let selection = Selection::new(TEN, 0, 0, None, 4 * 8);
        assert_selection!(selection, Ok(Selection::AlignedBinary(&[1, 1, 1, 1])));

        // Selecting a subset of bytes, from a byte-aligned offset, is also an aligned binary
        let selection = Selection::new(TEN, 9, 0, None, 8);
        assert_selection!(selection, Ok(Selection::AlignedBinary(&[1])));

        // Selecting too many bytes from an aligned binary still produces a fallback
        // selection that is an aligned binary
        let selection = Selection::new(TEN, 0, 0, None, 20 * 8);
        assert_selection!(selection, Err(Selection::AlignedBinary(TEN)));
    }

    #[test]
    fn selection_test_aligned_bitstring() {
        // Selecting a subset of bits containing at least one partial byte, from an aligned binary,
        // produces an aligned bitstring
        let buf = &[u8::MAX, u8::MAX, u8::MAX];
        let selection = Selection::new(buf, 0, 0, None, 14);
        assert_selection!(
            selection,
            Ok(Selection::AlignedBitstring(
                &[u8::MAX],
                MaybePartialByte::new(0b11111100, 6)
            ))
        );
        let selection = Selection::new(buf, 0, 0, None, 7);
        assert_selection!(
            selection,
            Ok(Selection::Byte(MaybePartialByte::new(0b11111110, 7)))
        );

        // Selecting beyond the selectable range, when the selectable range constitutes an aligned
        // bitstring, produces an aligned bitstring
        let selection = Selection::new(buf, 1, 0, Some(23), 24);
        assert_selection!(
            selection,
            Err(Selection::AlignedBitstring(
                &[u8::MAX],
                MaybePartialByte::new(0b11111110, 7),
            ))
        );
    }

    #[test]
    fn selection_test_binary() {
        // Selecting a byte-divisible number of bits from any non-byte aligned offset produces a
        // binary selection
        let selection = Selection::new(TEN, 0, 1, None, 2 * 8);
        assert_selection!(
            selection,
            Ok(Selection::Binary(
                MaybePartialByte::new(2, 7),
                &[1],
                MaybePartialByte::new(0, 1)
            ))
        );

        let selection = Selection::new(TEN, 0, 1, None, 4 * 8);
        assert_selection!(
            selection,
            Ok(Selection::Binary(
                MaybePartialByte::new(2, 7),
                &[1, 1, 1],
                MaybePartialByte::new(0, 1)
            ))
        );

        // Selecting beyond the selectable range when the selectable range constitutes an unaligned
        // binary, produces an unaligned binary selection
        let selection = Selection::new(TEN, 0, 1, Some(17), 24);
        assert_selection!(
            selection,
            Err(Selection::Binary(
                MaybePartialByte::new(2, 7),
                &[1],
                MaybePartialByte::new(0, 1),
            ))
        );
    }

    #[test]
    fn selection_test_bitstring() {
        // Selecting any number of bits (> 8, non-byte-divisible) when unaligned, produces an
        // unaligned bitstring
        let buf = &[u8::MAX, u8::MAX, u8::MAX];
        let selection = Selection::new(buf, 0, 3, None, 17);
        assert_selection!(
            selection,
            Ok(Selection::Bitstring(
                MaybePartialByte::new(0b11111000, 5),
                &[u8::MAX],
                Some(MaybePartialByte::new(0b11110000, 4))
            ))
        );

        // Selecting beyond the selectable range when the selectable range constitutes an unaligned
        // bitstring, produces an unaligned bitstring
        let selection = Selection::new(buf, 0, 3, None, 27);
        assert_selection!(
            selection,
            Err(Selection::Bitstring(
                MaybePartialByte::new(0b11111000, 5),
                &[u8::MAX, u8::MAX],
                None,
            ))
        );

        // Selecting a number of bits from an unaligned bit that ends on a byte boundary produces a
        // bitstring
        let selection = Selection::new(buf, 0, 7, None, 9);
        assert_selection!(
            selection,
            Ok(Selection::Bitstring(
                MaybePartialByte::new(0b10000000, 1),
                &[u8::MAX],
                None,
            ))
        );

        // Non-fitting, non-aligned bitstring which is too big by more than 8 bits
        let selection = Selection::new(buf, 1, 2, Some(23), 24);
        assert_selection!(
            selection,
            Err(Selection::Bitstring(
                MaybePartialByte::new(0b11111100, 6),
                &[],
                Some(MaybePartialByte::new(0b11111110, 7))
            ))
        );

        let buf = &[u8::MAX, u8::MAX, u8::MAX, u8::MAX];
        let selection = Selection::new(buf, 1, 2, Some(31), 32);
        assert_selection!(
            selection,
            Err(Selection::Bitstring(
                MaybePartialByte::new(0b11111100, 6),
                &[u8::MAX],
                Some(MaybePartialByte::new(0b11111110, 7))
            ))
        );

        let selection = Selection::new(HELLO, 1, 2, None, 30);
        assert_selection!(
            selection,
            Ok(Selection::Bitstring(
                MaybePartialByte::new(0b10010100, 6),
                &HELLO[2..],
                None,
            ))
        );
    }

    #[test]
    fn selection_test_get() {
        let slice = Selection::new(HELLO, 0, 1, None, 8).unwrap();
        assert_eq!(slice.get(0), Some(0b11010000));
    }

    #[test]
    fn selection_test_pop() {
        // Empty
        let mut slice = Selection::new(&[], 0, 0, None, 0).unwrap();
        assert_eq!(slice.pop(), None);

        // Byte
        let mut slice = Selection::new(HELLO, 4, 0, None, 8).unwrap();
        assert_eq!(slice.pop(), Some('o' as u8));
        assert_eq!(slice.pop(), None);

        // Partial byte
        let buf = &[u8::MAX, u8::MAX, u8::MAX];
        let mut slice = Selection::new(buf, 2, 1, None, 7).unwrap();
        assert_eq!(slice.pop(), Some(0b11111110));
        assert_eq!(slice.pop(), None);

        // Aligned binary
        let mut slice = Selection::new(HELLO, 0, 0, None, 32).unwrap();
        assert_eq!(slice.pop(), Some(104));
        assert_eq!(slice.pop(), Some(101));
        assert_eq!(slice.pop(), Some(108));
        assert_eq!(slice.pop(), Some(108));
        assert_eq!(slice.pop(), None);

        // Unaligned binary
        // 01101000 01100101 01101100 01101100 01101111
        //  ^------ -*------ -*------ -*------ ^
        // 0        1        2        3        4
        let mut slice = Selection::new(HELLO, 0, 1, None, 32).unwrap();
        assert_eq!(
            ExactSelection(slice),
            ExactSelection(Selection::Binary(
                MaybePartialByte::new(0b11010000, 7),
                &HELLO[1..4],
                MaybePartialByte::new(0b00000000, 1)
            ))
        );
        assert_eq!(slice.pop(), Some(0b11010000));
        assert_eq!(slice.pop(), Some(0b11001010));
        assert_eq!(slice.pop(), Some(0b11011000));
        assert_eq!(slice.pop(), Some(0b11011000));
        assert_eq!(slice.pop(), None);

        // Aligned bitstring
        // 01101000 01100101 01101100 01101100 01101111
        // ^------- *------- *------- *--^
        // 0        1        2        3        4
        let mut slice = Selection::new(HELLO, 0, 0, None, 28).unwrap();
        assert_eq!(slice.pop(), Some('h' as u8));
        assert_eq!(slice.pop(), Some('e' as u8));
        assert_eq!(slice.pop(), Some('l' as u8));
        assert_eq!(slice.pop(), Some(0b01100000));
        assert_eq!(slice.pop(), None);

        // Unaligned bitstring
        // 01101000 01100101 01101100 01101100 01101111
        //  ^------ -*------ -*------ -*---^
        // 0        1        2        3        4
        let mut slice = Selection::new(HELLO, 0, 1, None, 28).unwrap();
        assert_eq!(slice.pop(), Some(0b11010000));
        assert_eq!(slice.pop(), Some(0b11001010));
        assert_eq!(slice.pop(), Some(0b11011000));
        assert_eq!(slice.pop(), Some(0b11010000));
        assert_eq!(slice.pop(), None);
    }

    #[test]
    fn selection_test_shrink_front() {
        let selection = Selection::all(HELLO);

        // Shrinking an aligned binary by a number of bits divisible into bytes produces an aligned
        // binary
        let selection = selection.shrink_front(8);
        assert_eq!(
            ExactSelection(selection),
            ExactSelection(Selection::AlignedBinary(&HELLO[1..]))
        );

        // Shrinking an aligned binary by a odd number of bits produces an unaligned bitstring
        let selection = selection.shrink_front(1);
        assert_eq!(
            ExactSelection(selection),
            ExactSelection(Selection::Bitstring(
                MaybePartialByte::new(0b11001010, 7),
                &HELLO[2..],
                None,
            ))
        );

        // Shrinking an unaligned bitstring by a number of bits that ends on a byte boundary
        // produces an aligned binary
        let selection2 = selection.shrink_front(7);
        assert_eq!(
            ExactSelection(selection2),
            ExactSelection(Selection::AlignedBinary(&HELLO[2..])),
        );

        // Shrinking an unaligned bitstring by a binary number of bits produces an unaligned binary
        let selection3 = Selection::new(HELLO, 0, 1, None, 32).unwrap();
        assert_eq!(selection3.bit_size(), 32);
        assert_eq!(
            ExactSelection(selection3),
            ExactSelection(Selection::Binary(
                MaybePartialByte::new(0b11010000, 7),
                &HELLO[1..4],
                MaybePartialByte::new(0b00000000, 1),
            ))
        );
        let selection3 = selection3.shrink_front(8);
        assert_eq!(
            ExactSelection(selection3),
            ExactSelection(Selection::Binary(
                MaybePartialByte::new(0b11001010, 7),
                &HELLO[2..4],
                MaybePartialByte::new(0b00000000, 1)
            )),
        );

        // Shrinking a selection to 7 bits or less produces a byte
        let selection = Selection::all(HELLO);
        let selection = selection.shrink_front(33);
        assert_eq!(
            ExactSelection(selection),
            ExactSelection(Selection::Byte(MaybePartialByte::new(0b11011110, 7)))
        );
    }

    #[test]
    fn selection_test_take() {
        let selection = Selection::all(HELLO);

        // Taking a byte produces a new selection of one byte
        let selection1 = selection.take(8);
        assert_selection!(selection1, Ok(Selection::AlignedBinary(&HELLO[0..1])));

        // Taking less than a bytes worth of bits produces a byte selection of appropriate size
        let selection2 = selection.take(1);
        assert_selection!(selection2, Ok(Selection::Byte(MaybePartialByte::new(0, 1))));

        let selection2 = selection.take(2);
        assert_selection!(
            selection2,
            Ok(Selection::Byte(MaybePartialByte::new(0b01000000, 2)))
        );

        let selection2 = selection.take(7);
        assert_selection!(
            selection2,
            Ok(Selection::Byte(MaybePartialByte::new(0b01101000, 7)))
        );

        // Taking a non-binary number of bits from an aligned selection produces an aligned
        // bitstring
        let selection3 = selection.take(9);
        assert_selection!(
            selection3,
            Ok(Selection::AlignedBitstring(
                &HELLO[0..1],
                MaybePartialByte::new(0, 1)
            ))
        );

        // Taking a byte with only bits remaining produces a partial byte
        let selection = Selection::new(HELLO, 0, 7, None, 6).unwrap();
        assert_eq!(
            ExactSelection(selection),
            ExactSelection(Selection::Byte(MaybePartialByte::new(0b00110000, 6)))
        );

        let selection4 = selection.take(8);
        assert_selection!(
            selection4,
            Err(Selection::Byte(MaybePartialByte::new(0b00110000, 6)))
        );

        // Taking more than a byte of bits with only < 8 bits remaining produces a partial byte
        let selection4 = selection.take(20);
        assert_selection!(
            selection4,
            Err(Selection::Byte(MaybePartialByte::new(0b00110000, 6)))
        );

        let buf = &[u8::MAX, u8::MAX, u8::MAX];
        // Select 4 bits in the middle of the second byte
        let selection = Selection::new(buf, 1, 3, Some(15), 4);
        assert_selection!(
            selection,
            Ok(Selection::Byte(MaybePartialByte::new(0b11110000, 4)))
        );
    }
}

use alloc::alloc::{Allocator, Global};
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::Extend;
use core::mem;

use num_bigint::BigInt;

use super::*;

/// This struct facilitates construction of bitstrings/binaries by providing
/// a resizable buffer into which data can be written using one of several convenience
/// functions, or Rust's formatting facilities as both `std::fmt::Write` and `std::io::Write`
/// are implemented for it.
///
/// This struct also implements `Bitstring`, and so may be used anywhere that bitstrings are
/// allowed.
///
/// # Example
///
/// ```ignore
/// let mut w = BitVec::new();
/// write!(&mut w, "Hello, {}!", name)?;
/// ```
#[derive(Clone)]
pub struct BitVec<A: Allocator = Global> {
    data: Vec<u8, A>,
    pos: usize,
    bit_offset: u8,
}
impl<A: Allocator> fmt::Debug for BitVec<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BitVec")
            .field("data", &self.data.as_slice())
            .field("capacity", &self.data.capacity())
            .field("len", &self.data.len())
            .field("pos", &self.pos)
            .field("bit_offset", &self.bit_offset)
            .finish()
    }
}
impl<A: Allocator> From<Vec<u8, A>> for BitVec<A> {
    fn from(data: Vec<u8, A>) -> Self {
        let pos = if data.is_empty() { 0 } else { data.len() - 1 };
        Self {
            data,
            pos,
            bit_offset: 0,
        }
    }
}
impl Default for BitVec {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            pos: 0,
            bit_offset: 0,
        }
    }
}
impl BitVec {
    /// Create a new, empty BitVec
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new BitVec with the given initial capacity
    pub fn with_capacity(cap: usize) -> Self {
        let mut data = Vec::with_capacity(cap);
        unsafe {
            data.set_len(data.capacity());
        }
        Self {
            data,
            pos: 0,
            bit_offset: 0,
        }
    }
}
impl<A: Allocator> BitVec<A> {
    /// Convert a Vec<u8, A> + bit offset into a BitVec<A>
    pub fn from_vec_with_trailing_bits(data: Vec<u8, A>, bit_offset: u8) -> Self {
        assert!(bit_offset < 8);
        let pos = if data.is_empty() { 0 } else { data.len() - 1 };
        Self {
            data,
            pos,
            bit_offset,
        }
    }

    /// Create a new, empty BitVec using the given allocator
    pub fn new_in(alloc: A) -> Self {
        Self {
            data: Vec::new_in(alloc),
            pos: 0,
            bit_offset: 0,
        }
    }

    /// Create a new BitVec with the given initial capacity and allocator
    pub fn with_capacity_in(cap: usize, alloc: A) -> Self {
        let mut data = Vec::with_capacity_in(cap, alloc);
        unsafe {
            data.set_len(data.capacity());
        }
        Self {
            data,
            pos: 0,
            bit_offset: 0,
        }
    }

    /// Resets this BitVec to its initial state.
    ///
    /// NOTE: This does not change the allocated capacity of the underlying buffer
    pub fn clear(&mut self) {
        self.data.clear();
        self.pos = 0;
        self.bit_offset = 0;
    }

    /// Returns the byte at `index` in the underlying buffer
    pub fn get(&self, index: usize) -> Option<u8> {
        if index >= (self.pos + (self.bit_offset > 0) as usize) {
            return None;
        }
        Some(unsafe { *self.data.get_unchecked(index) })
    }

    /// Returns a selection covering the initialized region of the underlying buffer
    ///
    /// NOTE: As this takes a reference on the underlying buffer, it is not possible to
    /// obtain a slice while there are outstanding mutable references to this BitVec
    pub fn select(&self) -> Selection<'_> {
        Selection::new(self.data.as_slice(), 0, 0, None, self.bit_size()).unwrap()
    }

    /// Start a pattern match on this BitVec
    ///
    /// NOTE: This function takes a reference on the underlying buffer, so it is not possible
    /// to start a match while there are outstanding mutable references to this BitVec
    pub fn matcher(&self) -> Matcher<'_> {
        Matcher::new(self.select())
    }

    /// Returns the available capacity in bytes of the underlying buffer
    #[inline]
    fn bytes_available(&self) -> usize {
        self.data.capacity() - self.byte_size()
    }
}

impl<A: Allocator> Bitstring for BitVec<A> {
    fn byte_size(&self) -> usize {
        self.pos + ((self.bit_offset > 0) as usize)
    }

    fn bit_size(&self) -> usize {
        (self.pos * 8) + self.bit_offset as usize
    }

    #[inline]
    fn bit_offset(&self) -> u8 {
        self.bit_offset
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        &self.data[..self.byte_size()]
    }
}

impl<A: Allocator> BitVec<A> {
    /// Write a `Selection` via this writer
    pub fn push_selection(&mut self, selection: Selection<'_>) {
        match selection {
            Selection::Empty => (),
            Selection::Byte(b) if b.is_partial() => {
                unsafe { self.push_partial_byte(b.byte(), b.size) };
            }
            Selection::Byte(b) => {
                self.push_byte(b.byte());
            }
            Selection::AlignedBinary(bytes) => {
                self.push_bytes(bytes);
            }
            Selection::Binary(l, bytes, r) => {
                unsafe { self.push_partial_byte(l.byte(), l.size) };
                self.push_bytes(bytes);
                unsafe { self.push_partial_byte(r.byte(), r.size) };
            }
            Selection::AlignedBitstring(bytes, r) => {
                self.push_bytes(bytes);
                unsafe { self.push_partial_byte(r.byte(), r.size) };
            }
            Selection::Bitstring(l, bytes, None) => {
                unsafe { self.push_partial_byte(l.byte(), l.size) };
                self.push_bytes(bytes);
            }
            Selection::Bitstring(l, bytes, Some(r)) => {
                unsafe { self.push_partial_byte(l.byte(), l.size) };
                self.push_bytes(bytes);
                unsafe { self.push_partial_byte(r.byte(), r.size) };
            }
        }
    }

    /// Write a `str` via this writer
    #[inline]
    pub fn push_str(&mut self, s: &str) {
        self.push_bytes(s.as_bytes())
    }

    /// Write a single utf-8 `char` via this writer
    #[inline]
    pub fn push_utf8(&mut self, c: char) {
        let mut buf = [0; 4];
        self.push_str(c.encode_utf8(&mut buf));
    }

    /// Write a single utf-16 `char` via this writer
    #[inline]
    pub fn push_utf16(&mut self, c: char, endianness: Endianness) {
        let mut buf = [0; 2];
        let codepoints = c.encode_utf16(&mut buf);
        for cp in codepoints {
            self.push_number(*cp, endianness);
        }
    }

    /// Write a single utf-32 `char` via this writer
    #[inline]
    pub fn push_utf32(&mut self, c: char, endianness: Endianness) {
        self.push_number(c as u32, endianness)
    }

    /// Write a single numeric value via this writer
    ///
    /// NOTE: The provided value will be converted to the specified endianness, so you
    /// should make sure that the value has not already been manually converted, or the
    /// written value will be different than expected.
    pub fn push_number<N, const S: usize>(&mut self, n: N, endianness: Endianness)
    where
        N: ToEndianBytes<S>,
    {
        match endianness {
            Endianness::Native => self.push_bytes(&n.to_ne_bytes()),
            Endianness::Big => self.push_bytes(&n.to_be_bytes()),
            Endianness::Little => self.push_bytes(&n.to_le_bytes()),
        }
    }

    /// Like `push_number`, but with a specific arbitrary precision.
    ///
    /// If you want to push a number with the exact precision of its type, use `push_number`.
    ///
    /// This function will take `bitsize` bits from the byte representation of `n` in the specified
    /// endian order, until either the specified number of bits have been written, or there are no
    /// more bits in the provided value, at which point the remaining bits are written as zeroes.
    ///
    /// When this function returns, `bitsize` bits will always have been written to the buffer, no
    /// more, no less.
    pub fn push_ap_number<N, const S: usize>(
        &mut self,
        n: N,
        bitsize: usize,
        endianness: Endianness,
    ) where
        N: ToEndianBytes<S>,
    {
        assert!(bitsize <= S * 8);
        let padding_bits = bitsize % 8;
        let padding_bytes = S - ((bitsize / 8) + (padding_bits > 0) as usize);
        let bytes = match endianness {
            Endianness::Native => n.to_ne_bytes(),
            Endianness::Big => n.to_be_bytes(),
            Endianness::Little => n.to_le_bytes(),
        };
        let mut bytes = bytes.as_slice();
        self.push_bits(bytes.take(padding_bytes..).unwrap(), bitsize)
    }

    /// Writes a big integer value to the buffer in the specified endianness.
    pub fn push_bigint(&mut self, i: &BigInt, signed: bool, endianness: Endianness) {
        let bytes = match endianness {
            Endianness::Native if signed => {
                if cfg!(target_endian = "big") {
                    i.to_signed_bytes_be()
                } else {
                    i.to_signed_bytes_le()
                }
            }
            Endianness::Native => {
                if cfg!(target_endian = "big") {
                    i.to_bytes_be().1
                } else {
                    i.to_bytes_le().1
                }
            }
            Endianness::Big if signed => i.to_signed_bytes_be(),
            Endianness::Big => i.to_bytes_be().1,
            Endianness::Little if signed => i.to_signed_bytes_le(),
            Endianness::Little => i.to_bytes_le().1,
        };
        self.push_bytes(bytes.as_slice());
    }

    /// Like `push_ap_number`, but for big integer values.
    pub fn push_ap_bigint(
        &mut self,
        i: &BigInt,
        bitsize: usize,
        signed: bool,
        endianness: Endianness,
    ) {
        let bytes = match endianness {
            Endianness::Native if signed => {
                if cfg!(target_endian = "big") {
                    i.to_signed_bytes_be()
                } else {
                    i.to_signed_bytes_le()
                }
            }
            Endianness::Native => {
                if cfg!(target_endian = "big") {
                    i.to_bytes_be().1
                } else {
                    i.to_bytes_le().1
                }
            }
            Endianness::Big if signed => i.to_signed_bytes_be(),
            Endianness::Big => i.to_bytes_be().1,
            Endianness::Little if signed => i.to_signed_bytes_le(),
            Endianness::Little => i.to_bytes_le().1,
        };
        self.push_bits(bytes.as_slice(), bitsize)
    }

    /// Write a single bit via this writer
    #[inline]
    pub fn push_bit(&mut self, bit: bool) {
        self.reserve(1);

        // Fast path
        unsafe {
            if self.bit_offset == 0 {
                self.push_bit_fast(bit);
            } else {
                self.push_bit_slow(bit);
            }
        }
    }

    /// Used to push a bit when the current position is byte-aligned
    #[inline]
    unsafe fn push_bit_fast(&mut self, bit: bool) {
        let ptr = self.data.as_mut_ptr();
        *ptr.add(self.pos) = (bit as u8) << 7;
        self.bit_offset += 1;
    }

    /// Used to push a bit when the current position is NOT byte-aligned
    unsafe fn push_bit_slow(&mut self, bit: bool) {
        // First, we need to rewrite the current partial byte with bits from `byte`
        // Then, we need to shift the remaining bits of `byte` left and write that as a new partial
        // byte
        let ptr = self.data.as_mut_ptr();
        let partial_byte = ptr.add(self.pos);

        // This mask extracts the bits which need to be shifted left to form the new partial byte
        // When inverted, it extracts the bits which will fill out the current partial byte
        let offset = self.bit_offset;
        let mask = (-1i8 as u8) >> offset;

        // Rewrite the partial byte
        let left_bits = *partial_byte & !mask;
        let right_bits = (bit as u8) << (7 - offset);
        *partial_byte = left_bits | right_bits;

        // We shift our position forward one byte, the bit offset remains unchanged
        if offset == 7 {
            // We've filled out the partial byte, so we need to increment our byte position
            self.pos += 1;
            self.bit_offset = 0;
        } else {
            self.bit_offset += 1;
        }
    }

    /// Write a single byte via this writer
    #[inline]
    pub fn push_byte(&mut self, byte: u8) {
        self.reserve(1);

        // Fast path
        unsafe {
            if self.bit_offset == 0 {
                self.push_byte_fast(byte);
            } else {
                self.push_byte_slow(byte);
            }
        }
    }

    /// Used to push a byte when the current position is byte-aligned
    #[inline]
    unsafe fn push_byte_fast(&mut self, byte: u8) {
        let ptr = self.data.as_mut_ptr();
        *ptr.add(self.pos) = byte;
        self.pos += 1;
    }

    /// Used to push a byte when the current position is NOT byte-aligned
    #[cold]
    unsafe fn push_byte_slow(&mut self, byte: u8) {
        // First, we need to rewrite the current partial byte with bits from `byte`
        // Then, we need to shift the remaining bits of `byte` left and write that as a new partial
        // byte
        let ptr = self.data.as_mut_ptr();
        let partial_byte = ptr.add(self.pos);

        // This mask extracts the bits which need to be shifted left to form the new partial byte
        // When inverted, it extracts the bits which will fill out the current partial byte
        let offset = self.bit_offset;
        let mask = !(u8::MAX >> offset);

        // Rewrite the partial byte
        let left_bits = *partial_byte & mask;
        let right_bits = byte >> offset;
        *partial_byte = left_bits | right_bits;

        // We shift our position forward one byte, the bit offset remains unchanged
        self.pos += 1;

        // Mask out the remaining bits and shift them left by the bit offset to form the new partial
        // byte
        *ptr.add(self.pos) = byte << (8 - offset);
    }

    /// Concatenantes the contents of another BitVec to this one
    pub fn concat(&mut self, other: &Self) {
        if other.bit_offset == 0 {
            self.push_bytes(unsafe { other.as_bytes_unchecked() })
        } else {
            let bytes = unsafe { other.as_bytes_unchecked() };
            self.push_bits(bytes, other.bit_size())
        }
    }

    /// Write a slice of bytes via this writer
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        if len == 0 {
            return;
        }

        self.reserve(len);

        if self.bit_offset == 0 {
            unsafe {
                self.push_bytes_fast(bytes);
            }
        } else {
            unsafe {
                self.push_bytes_slow(bytes);
            }
        }
    }

    /// Writes exactly `size` bytes via this writer, starting by consuming as many bytes
    /// as possible from `bytes`, then writing zeros for any remaining bytes.
    pub fn push_bytes_exact(&mut self, bytes: &[u8], size: usize) {
        let available = bytes.len();
        if available >= size {
            self.push_bytes(&bytes[0..size]);
        } else {
            self.push_bytes(bytes);
            let padding = size - available;
            self.reserve(padding);
            let ptr = if self.bit_offset == 0 {
                unsafe { self.data.as_mut_ptr().add(self.pos) }
            } else {
                // We will write padding bytes starting at the next whole byte
                // Since partial bytes are always written with unused bits zeroed,
                // this has the same effect as if we were writing a byte slice of
                // all zeroes, and ensures the final partial byte is also zeroed
                // while preserving our bit offset so that the next bits to be written
                // will begin at the proper offset
                unsafe { self.data.as_mut_ptr().add(self.pos + 1) }
            };
            unsafe {
                core::ptr::write_bytes(ptr, 0, padding);
                self.pos += padding;
            }
        }
    }

    #[inline]
    unsafe fn push_bytes_fast(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        let src = bytes.as_ptr();
        let dst = self.data.as_mut_ptr().add(self.pos);
        core::ptr::copy_nonoverlapping(src, dst, len);
        self.pos += len;
    }

    #[cold]
    unsafe fn push_bytes_slow(&mut self, bytes: &[u8]) {
        // We need to shift some bits from every byte into the previous byte
        // So we simply walk the byte slice and write bytes with `push_byte_slow`
        for byte in bytes {
            self.push_byte_slow(*byte);
        }
    }

    #[inline]
    fn reserve(&mut self, cap: usize) {
        let available = self.bytes_available();
        if available >= cap {
            return;
        }
        // Buffer some additional capacity above and beyond
        let cap = mem::size_of::<usize>() + cap;
        self.data.reserve(cap - available);
        unsafe {
            self.data.set_len(self.data.capacity());
        }
    }

    /// Write exactly `size` bits to the buffer, starting by consuming as many bits
    /// as possible from `bytes`, then writing zeros for any remaining bits.
    pub fn push_bits(&mut self, bytes: &[u8], size: usize) {
        let trailing_bits = size % 8;
        if trailing_bits == 0 {
            return self.push_bytes_exact(bytes, size / 8);
        }

        let available_bytes = bytes.len();
        let available = available_bytes * 8;
        if available >= size {
            // We can get all bits from the buffer
            self.push_bits_exact(bytes, size);
        } else {
            // We require some padding bits at the end
            self.push_bytes(bytes);
            // Calculate the number of remaining bits to write
            let remaining_bits = size - available;
            // If after pushing the bits that were available, we are aligned on a byte boundary, it
            // vastly simplifies writing the padding bytes and handling the trailing
            // bits
            if self.bit_offset == 0 {
                // Recalculate the number of trailing bits
                let trailing_bits = (remaining_bits % 8) as u8;
                let remaining_bytes = remaining_bits / 8;
                let padding = remaining_bytes + (trailing_bits > 0) as usize;
                self.reserve(padding);
                unsafe {
                    let ptr = self.data.as_mut_ptr().add(self.pos);
                    core::ptr::write_bytes(ptr, 0, padding);
                }
                self.pos += remaining_bytes;
                self.bit_offset = trailing_bits;
                return;
            }

            // Otherwise, we need to subtract the remaining bits of the current partial byte
            // from the number of bits remaining, and recalculate the trailing bits
            let partial_bits = 8 - self.bit_offset;
            // If the number of padding bits would fit in the remaining bits of
            // the partial byte, simply adjust the bit offset and we're done as
            // those bits have already been zeroed
            if remaining_bits < partial_bits as usize {
                self.bit_offset += remaining_bits as u8;
                return;
            }

            let remaining_bits = remaining_bits - partial_bits as usize;
            let trailing_bits = (remaining_bits % 8) as u8;
            let remaining_bytes = remaining_bits / 8;
            let padding = remaining_bytes + (trailing_bits > 0) as usize;
            self.reserve(padding);
            unsafe {
                let ptr = self.data.as_mut_ptr().add(self.pos + 1);
                core::ptr::write_bytes(ptr, 0, padding);
            }
            self.pos += remaining_bytes;
            self.bit_offset = trailing_bits;
        }
    }

    // This function is used when we can get all of the bits we need from the input slice
    //
    // We do this by first pushing all of the bytes except the last one as a slice, in order
    // to benefit from optimizations available when performing aligned writes; then the remaining
    // bits are handled manually depending on the bit offset and number of remaining bits.
    fn push_bits_exact(&mut self, bytes: &[u8], size: usize) {
        if size == 0 {
            return;
        }

        let byte_size = size / 8;
        let trailing_bits = (size % 8) as u8;

        // If the number of bits requested fits in a single byte, we can proceed directly to
        // handling the final byte
        if byte_size == 0 {
            return unsafe { self.push_partial_byte(bytes[0], trailing_bits) };
        }

        // If the number of bits requested is an evenly divisble number of bytes, we can delegate to
        // push_bytes
        if trailing_bits == 0 {
            return self.push_bytes(&bytes[0..byte_size]);
        }

        // Otherwise, if the current position is aligned, we can write all of the bytes
        // up to the last one, and handle it independently of any other bytes
        self.push_bytes(&bytes[0..byte_size]);
        unsafe {
            self.push_partial_byte(bytes[byte_size], trailing_bits);
        }
    }

    // In this case, there may be bits across two bytes that we need to combine
    //
    // For example, let's say we're writing 30 bits, and the initial bit offset was
    // 3, and we have a buffer of 4 bytes. This would give us a `trailing_bits` value
    // of 6. We would write the first 24 bits from the buffer, leaving us with 6 bits
    // remaining to write, with our offset unchanged, telling us that the byte `pos`
    // points to has 5 bits left that are unused. So we do the following:
    //
    // 1. Read the current partial byte pointed to by `pos`
    // 2. Read the final byte from the buffer
    // 3. Mask out the first 5 bits of the final byte and shift them right by `bit_offset`
    // 4. Bitwise-OR the value from #1 with the value of #3, this is the new value of the byte
    // pointed to by `pos`
    // 5. Mask out just the 6th bit of the final byte and shift it left by `8 - bit_offset`,
    // this is the new value of the byte pointed to by `pos + 1`
    unsafe fn push_partial_byte(&mut self, byte: u8, size: u8) {
        debug_assert!(size < 8);
        debug_assert_ne!(size, 0);
        // Discard the bits from byte which are not desired
        let byte = byte & (u8::MAX << (8 - size));
        // Get the current partial byte
        let ptr = self.data.as_mut_ptr().add(self.pos);
        let partial_byte = *ptr;
        // Calculate the shift needed to move bits into their appropriate locations
        // The shift corresponds to the number of unfilled bits in the partial byte
        let offset = self.bit_offset;
        let offset_shift = 8 - offset;
        // If this byte is aligned, we only need to write `size` bits of byte
        if offset_shift == 8 {
            let mask = u8::MAX << (8 - size);
            *ptr = byte & mask;
            self.bit_offset = size;
            return;
        }
        // Otherwise, mask out the bits for the partial byte and shift them into position, then
        // write the filled partial byte The inverse of this mask will extract the trailing
        // bits
        let mask = u8::MAX << offset_shift;
        let partial_byte = partial_byte & mask;
        *ptr = partial_byte | (byte >> offset);
        *ptr.add(1) = byte << offset_shift;
        // If the number of bits pushed is less than the number of unfilled bits in the
        // partial byte, then the current position remains unchanged and only the bit offset
        // changes. However, if the number of bits pushed spilled over into the next byte, then
        // we update both the position and the offset
        if offset_shift > size {
            self.bit_offset += size;
        } else {
            self.pos += 1;
            self.bit_offset = size - offset_shift;
        }
    }

    /// This is the fallback implementation for `extend` for cases where we don't
    /// know the length of the iterator
    fn default_extend<I: Iterator<Item = u8>>(&mut self, iter: I) {
        for byte in iter {
            self.push_byte(byte);
        }
    }
}
impl<A: Allocator> Eq for BitVec<A> {}
impl<A: Allocator, T: ?Sized + Bitstring> PartialEq<T> for BitVec<A> {
    fn eq(&self, other: &T) -> bool {
        // An optimization: we can say for sure that if the sizes don't match,
        // the slices don't either.
        if self.bit_size() != other.bit_size() {
            return false;
        }

        // If both slices are aligned binaries, we can compare their data directly
        if self.is_binary() && other.is_aligned() && other.is_binary() {
            let x = unsafe { self.as_bytes_unchecked() };
            let y = unsafe { other.as_bytes_unchecked() };
            return x == y;
        }

        // Otherwise we must fall back to a byte-by-byte comparison
        self.bytes().eq(other.bytes())
    }
}
impl<A: Allocator> Ord for BitVec<A> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        if self.is_binary() && other.is_binary() {
            unsafe {
                let x = self.as_bytes_unchecked();
                let y = other.as_bytes_unchecked();
                x.cmp(y)
            }
        } else {
            self.bytes().cmp(other.bytes())
        }
    }
}
impl<A: Allocator, T: ?Sized + Bitstring> PartialOrd<T> for BitVec<A> {
    // We order bitstrings lexicographically
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        // Aligned binaries can be compared using the optimal built-in slice comparisons in the
        // standard lib
        if self.is_binary() && other.is_aligned() && other.is_binary() {
            unsafe {
                let x = self.as_bytes_unchecked();
                let y = other.as_bytes_unchecked();
                Some(x.cmp(y))
            }
        } else {
            // Otherwise we must comapre byte-by-byte
            Some(self.bytes().cmp(other.bytes()))
        }
    }
}
impl<A: Allocator> Hash for BitVec<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.is_binary() {
            Hash::hash_slice(unsafe { self.as_bytes_unchecked() }, state);
        } else {
            for byte in self.bytes() {
                byte.hash(state);
            }
        }
    }
}

impl<A: Allocator> fmt::Write for BitVec<A> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.push_utf8(c);
        Ok(())
    }
}
#[cfg(feature = "std")]
impl<A: Allocator> std::io::Write for BitVec<A> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.push_bytes(buf);
        Ok(buf.len())
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
        let len = bufs.iter().map(|b| b.len()).sum();
        self.reserve(len);

        for buf in bufs {
            self.push_bytes(buf);
        }

        Ok(len)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.push_bytes(buf);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Extend<u8> for BitVec {
    fn extend<T: IntoIterator<Item = u8>>(&mut self, iter: T) {
        <Self as SpecExtend<T::IntoIter>>::spec_extend(self, iter.into_iter())
    }

    #[inline]
    fn extend_one(&mut self, byte: u8) {
        self.push_byte(byte);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }
}

trait SpecExtend<I> {
    fn spec_extend(&mut self, iter: I);
}
impl<I: Iterator<Item = u8>> SpecExtend<I> for BitVec {
    default fn spec_extend(&mut self, iter: I) {
        self.default_extend(iter)
    }
}
impl<'a> SpecExtend<ByteIter<'a>> for BitVec {
    fn spec_extend(&mut self, iter: ByteIter<'a>) {
        match iter.as_slice() {
            Some(bytes) => self.push_bytes(bytes),
            None => {
                self.reserve(iter.len());
                if self.bit_offset == 0 {
                    for byte in iter {
                        unsafe {
                            self.push_byte_fast(byte);
                        }
                    }
                } else {
                    for byte in iter {
                        unsafe {
                            self.push_byte_slow(byte);
                        }
                    }
                }
            }
        }
    }
}
impl<'a> SpecExtend<BitsIter<'a>> for BitVec {
    fn spec_extend(&mut self, mut iter: BitsIter<'a>) {
        match iter.as_slice() {
            Some(bytes) => self.push_bytes(bytes),
            None => {
                self.reserve(iter.byte_size());
                if self.bit_offset == 0 {
                    loop {
                        match iter.next() {
                            Some(byte) => unsafe { self.push_byte_fast(byte) },
                            None => {
                                if let Some(b) = iter.consume() {
                                    unsafe {
                                        self.push_partial_byte(b.byte(), b.size);
                                    }
                                }
                                break;
                            }
                        }
                    }
                } else {
                    loop {
                        match iter.next() {
                            Some(byte) => unsafe { self.push_byte_slow(byte) },
                            None => {
                                if let Some(b) = iter.consume() {
                                    unsafe {
                                        self.push_partial_byte(b.byte(), b.size);
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use alloc::vec::Vec;
    use num_bigint::{BigInt, Sign};

    #[test]
    fn bitvec_push_byte() {
        let mut vec = BitVec::new();

        vec.push_byte(0b11011011);
        assert_eq!(vec.byte_size(), 1);
        assert_eq!(vec.bit_size(), 8);
        assert_eq!(vec.pos, 1);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(0b11011011));
        assert_eq!(vec.get(1), None);

        vec.push_byte(0b00011100);
        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 16);
        assert_eq!(vec.pos, 2);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(0b11011011));
        assert_eq!(vec.get(1), Some(0b00011100));
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn bitvec_push_byte_unaligned() {
        let mut vec = BitVec::new();

        // This will hit the fast path since it is an aligned write
        vec.push_bit(true);
        vec.push_byte(0b11011011);
        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 9);
        assert_eq!(vec.pos, 1);
        assert_eq!(vec.bit_offset, 1);
        assert_eq!(vec.get(0), Some(0b11101101));
        assert_eq!(vec.get(1), Some(0b10000000));
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn bitvec_push_bit() {
        let mut vec = BitVec::new();

        // This will hit the fast path since it is an aligned write
        vec.push_bit(true);
        assert_eq!(vec.byte_size(), 1);
        assert_eq!(vec.bit_size(), 1);
        assert_eq!(vec.pos, 0);
        assert_eq!(vec.bit_offset, 1);
        assert_eq!(vec.get(0), Some(0b10000000));
        assert_eq!(vec.get(1), None);

        // This will hit the slow path since it is an unaligned write
        vec.push_bit(true);
        assert_eq!(vec.byte_size(), 1);
        assert_eq!(vec.bit_size(), 2);
        assert_eq!(vec.pos, 0);
        assert_eq!(vec.bit_offset, 2);
        assert_eq!(vec.get(0), Some(0b11000000));
        assert_eq!(vec.get(1), None);

        vec.push_bit(false);
        vec.push_bit(true);
        assert_eq!(vec.byte_size(), 1);
        assert_eq!(vec.bit_size(), 4);
        assert_eq!(vec.pos, 0);
        assert_eq!(vec.bit_offset, 4);
        assert_eq!(vec.get(0), Some(0b11010000));
        assert_eq!(vec.get(1), None);

        vec.push_bit(true);
        vec.push_bit(true);
        vec.push_bit(true);
        vec.push_bit(true);
        assert_eq!(vec.byte_size(), 1);
        assert_eq!(vec.bit_size(), 8);
        assert_eq!(vec.pos, 1);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(0b11011111));
        assert_eq!(vec.get(1), None);
    }

    #[test]
    fn bitvec_push_number() {
        let mut vec = BitVec::new();

        vec.push_number(0xdeadbeefu32, Endianness::Little);
        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(0xef));
        assert_eq!(vec.get(1), Some(0xbe));
        assert_eq!(vec.get(2), Some(0xad));
        assert_eq!(vec.get(3), Some(0xde));
        assert_eq!(vec.get(4), None);

        vec.push_number(0xdeadbeefu32, Endianness::Big);
        assert_eq!(vec.byte_size(), 8);
        assert_eq!(vec.bit_size(), 64);
        assert_eq!(vec.pos, 8);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(0xef));
        assert_eq!(vec.get(1), Some(0xbe));
        assert_eq!(vec.get(2), Some(0xad));
        assert_eq!(vec.get(3), Some(0xde));
        assert_eq!(vec.get(4), Some(0xde));
        assert_eq!(vec.get(5), Some(0xad));
        assert_eq!(vec.get(6), Some(0xbe));
        assert_eq!(vec.get(7), Some(0xef));
        assert_eq!(vec.get(8), None);

        // Push a few bits to make the remaining pushes unaligned
        vec.push_bit(true);
        vec.push_bit(false);
        assert_eq!(vec.byte_size(), 9);
        assert_eq!(vec.bit_size(), 66);
        assert_eq!(vec.pos, 8);
        assert_eq!(vec.bit_offset, 2);
        assert_eq!(vec.get(8), Some(0b10000000u8));
        assert_eq!(vec.get(9), None);

        vec.push_number(0xf00u16, Endianness::Big);
        assert_eq!(vec.byte_size(), 11);
        assert_eq!(vec.bit_size(), 82);
        assert_eq!(vec.pos, 10);
        assert_eq!(vec.bit_offset, 2);
        assert_eq!(vec.get(8), Some(0b10000011u8));
        assert_eq!(vec.get(9), Some(0b11000000u8));
        assert_eq!(vec.get(10), Some(0b00000000u8));
        assert_eq!(vec.get(11), None);

        // Push an arbitrary-precision integer equivalent to u4
        vec.push_ap_number(0xf0u8, 4, Endianness::Big);
        assert_eq!(vec.byte_size(), 11);
        assert_eq!(vec.bit_size(), 86);
        assert_eq!(vec.pos, 10);
        assert_eq!(vec.bit_offset, 6);
        assert_eq!(vec.get(10), Some(0b00111100u8));
        assert_eq!(vec.get(11), None);
    }

    #[test]
    fn bitvec_push_utf8() {
        let mut vec = BitVec::new();

        vec.push_utf8('a');
        assert_eq!(vec.byte_size(), 1);
        assert_eq!(vec.bit_size(), 8);
        assert_eq!(vec.pos, 1);
        assert_eq!(vec.bit_offset, 0);

        assert_eq!(vec.get(0), Some(97));

        vec.clear();

        vec.push_utf8('ø');
        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 16);
        assert_eq!(vec.pos, 2);
        assert_eq!(vec.bit_offset, 0);

        vec.clear();

        vec.push_utf8('〦');
        assert_eq!(vec.byte_size(), 3);
        assert_eq!(vec.bit_size(), 24);
        assert_eq!(vec.pos, 3);
        assert_eq!(vec.bit_offset, 0);

        vec.clear();

        vec.push_utf8('😀');
        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);
    }

    #[test]
    fn bitvec_push_utf16() {
        let mut vec = BitVec::new();

        vec.push_utf16('a', Endianness::Native);
        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 16);
        assert_eq!(vec.pos, 2);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(97));

        vec.clear();

        vec.push_utf16('ø', Endianness::Native);
        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 16);
        assert_eq!(vec.pos, 2);
        assert_eq!(vec.bit_offset, 0);

        vec.clear();

        vec.push_utf16('〦', Endianness::Native);
        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 16);
        assert_eq!(vec.pos, 2);
        assert_eq!(vec.bit_offset, 0);

        vec.clear();

        vec.push_utf16('😀', Endianness::Native);
        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);
    }

    #[test]
    fn bitvec_push_utf32() {
        let mut vec = BitVec::new();

        vec.push_utf32('a', Endianness::Native);
        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(97));

        vec.clear();

        vec.push_utf32('ø', Endianness::Native);
        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);

        vec.clear();

        vec.push_utf32('〦', Endianness::Native);
        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);

        vec.clear();

        vec.push_utf32('😀', Endianness::Native);
        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);
    }

    #[test]
    fn bitvec_push_bytes() {
        let mut vec = BitVec::new();
        vec.push_bytes(b"hello");

        assert_eq!(vec.byte_size(), 5);
        assert_eq!(vec.bit_size(), 40);
        assert_eq!(vec.pos, 5);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(104));
        assert_eq!(vec.get(1), Some(101));
        assert_eq!(vec.get(2), Some(108));
        assert_eq!(vec.get(3), Some(108));
        assert_eq!(vec.get(4), Some(111));
        assert_eq!(vec.get(5), None);
    }

    #[test]
    fn bitvec_push_bytes_unaligned() {
        let mut vec = BitVec::new();

        vec.push_bit(true);
        vec.push_bit(false);
        vec.push_bytes(b"yep");
        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 26);
        assert_eq!(vec.pos, 3);
        assert_eq!(vec.bit_offset, 2);
        assert_eq!(vec.get(0), Some(0b10011110));
        assert_eq!(vec.get(1), Some(0b01011001));
        assert_eq!(vec.get(2), Some(0b01011100));
        assert_eq!(vec.get(3), Some(0b00000000));
        assert_eq!(vec.get(4), None);
    }

    #[test]
    fn bitvec_push_bytes_exact() {
        let mut vec = BitVec::new();
        vec.push_bytes_exact(b"hello", 5);

        assert_eq!(vec.byte_size(), 5);
        assert_eq!(vec.bit_size(), 40);
        assert_eq!(vec.pos, 5);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(104));
        assert_eq!(vec.get(1), Some(101));
        assert_eq!(vec.get(2), Some(108));
        assert_eq!(vec.get(3), Some(108));
        assert_eq!(vec.get(4), Some(111));
        assert_eq!(vec.get(5), None);
    }

    #[test]
    fn bitvec_push_bytes_exact_subset() {
        let mut vec = BitVec::new();
        vec.push_bytes_exact(b"hello", 4);

        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(104));
        assert_eq!(vec.get(1), Some(101));
        assert_eq!(vec.get(2), Some(108));
        assert_eq!(vec.get(3), Some(108));
        assert_eq!(vec.get(4), None);
    }

    #[test]
    fn bitvec_push_bytes_exact_padded() {
        let mut vec = BitVec::new();
        vec.push_bytes_exact(b"hi", 4);

        assert_eq!(vec.byte_size(), 4);
        assert_eq!(vec.bit_size(), 32);
        assert_eq!(vec.pos, 4);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(104));
        assert_eq!(vec.get(1), Some(105));
        assert_eq!(vec.get(2), Some(0));
        assert_eq!(vec.get(3), Some(0));
        assert_eq!(vec.get(4), None);
    }

    #[test]
    fn bitvec_push_bits() {
        let mut vec = BitVec::new();
        vec.push_bits(b"hello", 40);

        assert_eq!(vec.byte_size(), 5);
        assert_eq!(vec.bit_size(), 40);
        assert_eq!(vec.pos, 5);
        assert_eq!(vec.bit_offset, 0);
        assert_eq!(vec.get(0), Some(104));
        assert_eq!(vec.get(1), Some(101));
        assert_eq!(vec.get(2), Some(108));
        assert_eq!(vec.get(3), Some(108));
        assert_eq!(vec.get(4), Some(111));
        assert_eq!(vec.get(5), None);
    }

    #[test]
    fn bitvec_push_bits_subset() {
        let mut vec = BitVec::new();
        vec.push_bits(b"hello", 14);

        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 14);
        assert_eq!(vec.pos, 1);
        assert_eq!(vec.bit_offset, 6);
        assert_eq!(vec.get(0), Some(104));
        assert_eq!(vec.get(1), Some(0b01100100));
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn bitvec_push_bits_subset_unaligned() {
        let mut vec = BitVec::new();
        vec.push_bit(true);
        vec.push_bits(b"hello", 14);

        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 15);
        assert_eq!(vec.pos, 1);
        assert_eq!(vec.bit_offset, 7);
        assert_eq!(vec.get(0), Some(0b10110100));
        assert_eq!(vec.get(1), Some(0b00110010));
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn bitvec_push_bits_padded() {
        let mut vec = BitVec::new();
        vec.push_bits(b"h", 14);

        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 14);
        assert_eq!(vec.pos, 1);
        assert_eq!(vec.bit_offset, 6);
        assert_eq!(vec.get(0), Some(104));
        assert_eq!(vec.get(1), Some(0));
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn bitvec_push_bits_padded_unaligned() {
        let mut vec = BitVec::new();
        vec.push_bit(true);
        vec.push_bits(b"h", 14);

        assert_eq!(vec.byte_size(), 2);
        assert_eq!(vec.bit_size(), 15);
        assert_eq!(vec.pos, 1);
        assert_eq!(vec.bit_offset, 7);
        assert_eq!(vec.get(0), Some(0b10110100));
        assert_eq!(vec.get(1), Some(0));
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn bitvec_integration_test() {
        // We're aiming to test that we can create the following bitstring:
        //
        //     <<0xdeadbeef::big-integer-size(4)-unit(8), 2::integer-size(1)-unit(8),
        // 5::integer-size(4)-unit(8), "hello"::binary>>
        //
        // Which should be equivalent to the following 14 hex-encoded bytes:
        //
        //     0xde ad be ef 02 00 00 00 05 68 65 6c 6c 6f
        //
        // Equivalent to the integer value 4307232375490090818593385583
        //
        // Equivalent to the following struct:
        //
        //     struct T { header: u32, version: u8, size: u32, data: [u8; 5] }
        let mut vec = BitVec::new();

        let bytes = b"hello";
        vec.push_number(0xdeadbeefu32, Endianness::Big);
        vec.push_number(2u8, Endianness::Big);
        vec.push_number(bytes.len() as u32, Endianness::Big);
        vec.push_bytes(bytes.as_slice());

        assert_eq!(vec.byte_size(), 14);
        assert_eq!(vec.bit_size(), 14 * 8);

        // Validate construction
        {
            let raw = unsafe { vec.as_bytes_unchecked() };
            let mut expected_buf = Vec::new();
            expected_buf.extend_from_slice(&0xdeadbeefu32.to_be_bytes());
            expected_buf.push(2);
            expected_buf.extend_from_slice(&5u32.to_be_bytes());
            expected_buf.extend_from_slice(b"hello");

            let expected = BigInt::from_radix_be(Sign::Plus, expected_buf.as_slice(), 256).unwrap();
            let i = BigInt::from_radix_be(Sign::Plus, raw, 256).unwrap();
            assert_eq!(&i, &expected);

            let verify_bytes: &'static [u8] = &[
                0xde, 0xad, 0xbe, 0xef, 0x2, 0x0, 0x0, 0x0, 0x5, 0x68, 0x65, 0x6c, 0x6c, 0x6f,
            ];
            assert_eq!(raw, verify_bytes);

            let verify = BigInt::from_radix_be(Sign::Plus, verify_bytes, 256).unwrap();
            assert_eq!(&i, &verify);
        }

        // Validate destructuring
        let mut matcher = vec.matcher();
        let header: Option<u32> = matcher.match_number(Endianness::Big);
        assert_eq!(header, Some(0xdeadbeef));
        let version: Option<u8> = matcher.match_number(Endianness::Big);
        assert_eq!(version, Some(2));
        let size: Option<u32> = matcher.match_number(Endianness::Big);
        assert_eq!(size, Some(5));
        let data = matcher.match_binary().unwrap();
        assert_eq!(data, b"hello");
    }
}

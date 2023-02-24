use alloc::vec::Vec;
use num_bigint::{BigInt, Sign};

use super::*;

/// This struct is used to perform pattern matching against a bitstring/binary.
///
/// It operates on a captured Selection so that the internal machinery for traversing
/// bits/bytes can be handled in one place, and to keep this struct focused on the
/// pattern matching semantics.
#[derive(Clone)]
pub struct Matcher<'a> {
    pub selection: Selection<'a>,
}
impl<'a> Matcher<'a> {
    pub fn new(selection: Selection<'a>) -> Self {
        Self { selection }
    }

    pub fn with_slice(data: &'a [u8]) -> Self {
        Self {
            selection: Selection::all(data),
        }
    }

    #[inline]
    pub fn bit_size(&self) -> usize {
        self.selection.bit_size()
    }

    #[inline]
    pub fn byte_size(&self) -> usize {
        self.selection.byte_size()
    }

    /// Reads a single byte from the current position in the input without
    /// advancing the input.
    pub fn read_byte(&mut self) -> Option<u8> {
        if self.selection.bit_size() < 8 {
            return None;
        }

        self.selection.get(0)
    }

    /// Reads a constant number of bytes into the provided buffer without
    /// advancing the input.
    pub fn read_bytes<const N: usize>(&mut self, buf: &mut [u8; N]) -> bool {
        if self.selection.byte_size() < N {
            return false;
        }

        if let Some(slice) = self.selection.as_bytes() {
            buf.copy_from_slice(&slice[..N]);
            return true;
        }

        let ptr = buf.as_mut_ptr();
        let mut idx = 0;
        for byte in self.selection.bytes().take(N) {
            unsafe {
                *ptr.add(idx) = byte;
            }
            idx += 1;
        }
        true
    }

    /// Reads the specified number of bits into the provided buffer.
    ///
    /// If the bitsize given is larger than the provided buffer, this function will panic.
    ///
    /// If the number of bits requested was read, returns true, otherwise false.
    pub fn read_bits<const N: usize>(&mut self, buf: &mut [u8; N], bitsize: usize) -> bool {
        assert!(bitsize <= N * 8);

        if let Ok(selection) = self.selection.take(bitsize) {
            // Ensure the buffer is zeroed
            unsafe {
                core::ptr::write_bytes(buf.as_mut_ptr(), 0, N);
            }

            let mut iter = selection.bits();
            let mut index = 0;
            while let Some(b) = iter.next() {
                buf[index] = b;
                index += 1;
            }
            match iter.consume() {
                None => {
                    // We're ending on the exact size of the request, we're done
                    // NOTE: This should only be reachable if the bitsize was binary
                    assert_eq!(bitsize % 8, 0);
                    true
                }
                Some(b) => {
                    // We're ending on the exact size of the request, we're done
                    buf[index] = b.byte();
                    true
                }
            }
        } else {
            false
        }
    }

    /// Reads a single number from the current position in the input without
    /// advancing the input. This should be used to build matchers which are
    /// fallible and operate on numeric values.
    pub fn read_number<N, const S: usize>(&mut self, endianness: Endianness) -> Option<N>
    where
        N: FromEndianBytes<S>,
    {
        let mut bytes = [0u8; S];
        if self.read_bytes(&mut bytes) {
            match endianness {
                Endianness::Native => Some(N::from_ne_bytes(bytes)),
                Endianness::Little => Some(N::from_le_bytes(bytes)),
                Endianness::Big => Some(N::from_be_bytes(bytes)),
            }
        } else {
            None
        }
    }

    pub fn read_ap_number<N, const S: usize>(
        &mut self,
        bitsize: usize,
        endianness: Endianness,
    ) -> Option<N>
    where
        N: FromEndianBytes<S> + core::ops::Shr<u32, Output = N>,
    {
        assert!(
            bitsize <= (S * 8),
            "requested bit size is larger than that of the containing type"
        );

        let trailing_bits = bitsize % 8;
        let byte_size = (bitsize / 8) + ((trailing_bits > 0) as usize);
        let rotation = S - byte_size;

        let mut bytes = [0u8; S];
        if self.read_bits(&mut bytes, bitsize) {
            match endianness {
                // When native endianness is requested, no manipulation of the buffer is required
                Endianness::Native => Some(N::from_le_bytes(bytes)),
                Endianness::Big => {
                    if cfg!(target_endian = "big") {
                        Some(N::from_be_bytes(bytes))
                    } else if trailing_bits > 0 {
                        // On little-endian systems, big-endian numbers must
                        // be shifted to the right by the number of trailing bits
                        let n = N::from_be_bytes(bytes);
                        let shift = ((rotation * 8) + (8 - trailing_bits)) as u32;
                        Some(n >> shift)
                    } else {
                        // When there are no trailing bits we can simply shift the bytes
                        bytes.rotate_right(rotation);
                        Some(N::from_be_bytes(bytes))
                    }
                }
                Endianness::Little => {
                    if cfg!(target_endian = "little") {
                        Some(N::from_le_bytes(bytes))
                    } else if trailing_bits > 0 {
                        // On big-endian systems, little-endian numbers must
                        // be shifted to the right by the number of trailing bits
                        let n = N::from_le_bytes(bytes);
                        let shift = ((rotation * 8) + (8 - trailing_bits)) as u32;
                        Some(n >> shift)
                    } else {
                        // When there are no trailing bits we can simply shift the bytes
                        bytes.rotate_right(rotation);
                        Some(N::from_le_bytes(bytes))
                    }
                }
            }
        } else {
            None
        }
    }

    pub fn read_bigint(
        &mut self,
        bitsize: usize,
        signed: bool,
        endianness: Endianness,
    ) -> Option<BigInt> {
        let endianness = if endianness == Endianness::Native {
            if cfg!(target_endian = "big") {
                Endianness::Big
            } else {
                Endianness::Little
            }
        } else {
            endianness
        };
        if let Ok(selection) = self.selection.take(bitsize) {
            let bytes = selection.bytes().collect::<Vec<_>>();
            match endianness {
                Endianness::Big if signed => Some(BigInt::from_signed_bytes_be(bytes.as_slice())),
                Endianness::Big => Some(BigInt::from_bytes_be(Sign::Plus, bytes.as_slice())),
                Endianness::Little if signed => {
                    Some(BigInt::from_signed_bytes_le(bytes.as_slice()))
                }
                Endianness::Little => Some(BigInt::from_bytes_le(Sign::Plus, bytes.as_slice())),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    /// Matches any primitive numeric type from the input, advancing the input if successful
    pub fn match_number<N, const S: usize>(&mut self, endianness: Endianness) -> Option<N>
    where
        N: FromEndianBytes<S>,
    {
        let matched = self.read_number(endianness)?;

        self.selection = self.selection.shrink_front(S * 8);

        Some(matched)
    }

    /// Matches any primitive numeric type from the input, of arbitrary bit size up to the
    /// containing type size, advancing the input if successful
    pub fn match_ap_number<N, const S: usize>(
        &mut self,
        bitsize: usize,
        endianness: Endianness,
    ) -> Option<N>
    where
        N: FromEndianBytes<S> + core::ops::Shr<u32, Output = N>,
    {
        let matched = self.read_ap_number(bitsize, endianness)?;

        self.selection = self.selection.shrink_front(bitsize);

        Some(matched)
    }

    /// Matches a big integer from the input, advancing the input if successful
    pub fn match_bigint(
        &mut self,
        bitsize: usize,
        signed: bool,
        endianness: Endianness,
    ) -> Option<BigInt> {
        let matched = self.read_bigint(bitsize, signed, endianness)?;

        self.selection = self.selection.shrink_front(bitsize);

        Some(matched)
    }

    /// Matches a single utf-8 character from the input
    pub fn match_utf8(&mut self) -> Option<char> {
        use core::str::{next_code_point, utf8_char_width};

        let first = self.read_byte()?;
        match utf8_char_width(first) {
            1 => {
                let c = char::from_u32(first as u32)?;
                self.selection = self.selection.shrink_front(8);
                Some(c)
            }
            2 => {
                let mut bytes = [0; 2];
                if self.read_bytes(&mut bytes) {
                    let mut iter = bytes.iter();
                    let c = unsafe {
                        next_code_point(&mut iter).map(|ch| char::from_u32_unchecked(ch))
                    }?;
                    self.selection = self.selection.shrink_front(16);
                    Some(c)
                } else {
                    None
                }
            }
            3 => {
                let mut bytes = [0; 3];
                if self.read_bytes(&mut bytes) {
                    let mut iter = bytes.iter();
                    let c = unsafe {
                        next_code_point(&mut iter).map(|ch| char::from_u32_unchecked(ch))
                    }?;
                    self.selection = self.selection.shrink_front(24);
                    Some(c)
                } else {
                    None
                }
            }
            4 => {
                let mut bytes = [0; 4];
                if self.read_bytes(&mut bytes) {
                    let mut iter = bytes.iter();
                    let c = unsafe {
                        next_code_point(&mut iter).map(|ch| char::from_u32_unchecked(ch))
                    }?;
                    self.selection = self.selection.shrink_front(32);
                    Some(c)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Matches a single utf-16 character from the input
    pub fn match_utf16(&mut self, endianness: Endianness) -> Option<char> {
        let decoder = Utf16CodepointReader::new(Matcher::new(self.selection), endianness);
        let c = char::decode_utf16(decoder)
            .take(1)
            .fold(None, |_acc, c| c.ok())?;
        self.selection = self.selection.shrink_front(char::len_utf16(c) * 16);
        Some(c)
    }

    /// Matches a single utf-32 character from the input
    pub fn match_utf32(&mut self, endianness: Endianness) -> Option<char> {
        let raw: u32 = self.read_number(endianness)?;
        let c = char::from_u32(raw)?;
        self.selection = self.selection.shrink_front(32);
        Some(c)
    }

    /// Matches a subslice of `n` bytes from the input, advancing the input if successful
    pub fn match_bytes(&mut self, n: usize) -> Option<Selection<'a>> {
        let bit_size = n * 8;
        match self.selection.take(bit_size) {
            Ok(selection) => {
                self.selection = self.selection.shrink_front(bit_size);
                Some(selection)
            }
            Err(_) => None,
        }
    }

    /// Matches a subslice of `n` bits from the input, advancing the input if successful
    pub fn match_bits(&mut self, n: usize) -> Option<Selection<'a>> {
        match self.selection.take(n) {
            Ok(selection) => {
                self.selection = self.selection.shrink_front(n);
                Some(selection)
            }
            Err(_) => None,
        }
    }

    /// Matches the rest of the underlying data as long as it is valid binary data, i.e.
    /// the number of bits is evenly divisible into bytes.
    ///
    /// Since this function consumes the matcher itself, it must always be the last match performed
    pub fn match_binary(&mut self) -> Option<Selection<'a>> {
        if self.selection.is_binary() {
            Some(core::mem::replace(&mut self.selection, Selection::Empty))
        } else {
            None
        }
    }

    /// This operation always succeeds by returning what remains of the underlyings slice
    ///
    /// Since this function consumes the matcher itself, it must always be the last match performed
    pub fn match_any(&mut self) -> Selection<'a> {
        core::mem::replace(&mut self.selection, Selection::Empty)
    }
}

struct Utf16CodepointReader<'a> {
    matcher: Matcher<'a>,
    endianness: Endianness,
}
impl<'a> Utf16CodepointReader<'a> {
    fn new(matcher: Matcher<'a>, endianness: Endianness) -> Self {
        Self {
            matcher,
            endianness,
        }
    }
}
impl<'a> Iterator for Utf16CodepointReader<'a> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.matcher.match_number(self.endianness)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn matcher_test_integers() {
        let bytes = 0xdeadbeefu32.to_be_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u32> = matcher.match_number(Endianness::Big);
        assert_eq!(num, Some(0xdeadbeef));
    }

    #[test]
    fn matcher_test_integer_endianness() {
        // big-endian 123456789u32 in a 64-bit buffer
        let bytes = [7, 91, 205, 21, 0, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(32, Endianness::Big);
        assert_eq!(num, Some(123456789u64));

        // little-endian 123456789u32 in a 64-bit buffer
        let bytes = [21, 205, 91, 7, 0, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(32, Endianness::Little);
        assert_eq!(num, Some(123456789u64));
    }

    #[test]
    fn matcher_test_ap_integer_oversized_buffer() {
        // 8 bits: we expect to pull out the number 7
        let bytes = [7, 91, 205, 21, 0, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(8, Endianness::Big);
        assert_eq!(num, Some(7u64));

        // 12 bits: we expect to pull out the number 117
        let bytes = [7, 91, 205, 21, 0, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(12, Endianness::Big);
        assert_eq!(num, Some(117u64));

        // 16 bits: we expect to pull out the number 1883
        let bytes = [7, 91, 205, 21, 0, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(16, Endianness::Big);
        assert_eq!(num, Some(1883u64));

        // 24 bits: we expect to pull out the number 482253
        let bytes = [7, 91, 205, 21, 0, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(24, Endianness::Big);
        assert_eq!(num, Some(482253u64));

        // 32 bits: we expect to pull out the number 123456789
        let bytes = [7, 91, 205, 21, 0, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(32, Endianness::Big);
        assert_eq!(num, Some(123456789u64));

        // 38 bits: we expect to pull out the number 7901234496
        let bytes = [7, 91, 205, 21, 1, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(38, Endianness::Big);
        assert_eq!(num, Some(7901234496u64));

        // 40 bits: we expect to pull out the number 31604937985
        let bytes = [7, 91, 205, 21, 1, 0, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(40, Endianness::Big);
        assert_eq!(num, Some(31604937985u64));

        // 48 bits: we expect to pull out the number 8090864124161
        let bytes = [7, 91, 205, 21, 1, 1, 0, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(48, Endianness::Big);
        assert_eq!(num, Some(8090864124161u64));

        // 56 bits: we expect to pull out the number 2071261215785217
        let bytes = [7, 91, 205, 21, 1, 1, 1, 0];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(56, Endianness::Big);
        assert_eq!(num, Some(2071261215785217u64));

        // 64 bits: we expect to pull out the number 530242871241015553
        let bytes = [7, 91, 205, 21, 1, 1, 1, 1];
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<u64> = matcher.match_ap_number(64, Endianness::Big);
        assert_eq!(num, Some(530242871241015553u64));
    }

    #[test]
    fn matcher_test_f16() {
        use half::f16;

        let f = f16::from_f64(1.0);
        let bytes = f.to_be_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f16> = matcher.match_number(Endianness::Big);
        assert_eq!(num, Some(f));

        let f = f16::from_f64(1.0);
        let bytes = f.to_le_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f16> = matcher.match_number(Endianness::Little);
        assert_eq!(num, Some(f));

        // Too large to fit in f16, should become infinite
        let f = f16::from_f64(f32::MAX as f64);
        assert!(f.is_infinite());
        let bytes = f.to_be_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f16> = matcher.match_number(Endianness::Big);
        assert_eq!(num, Some(f));
    }

    #[test]
    fn matcher_test_f32() {
        let bytes = 1.0f32.to_be_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f32> = matcher.match_number(Endianness::Big);
        assert_eq!(num, Some(1.0));

        let bytes = 1.0f32.to_le_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f32> = matcher.match_number(Endianness::Little);
        assert_eq!(num, Some(1.0));

        // Too large to fit in f32, should become infinite
        let bytes = (1.0_f32 / 0.0_f32).to_be_bytes();
        let f = f32::from_be_bytes(bytes);
        assert!(f.is_infinite());
        let bytes = f.to_be_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f32> = matcher.match_number(Endianness::Big);
        assert_eq!(num, Some(f));
    }

    #[test]
    fn matcher_test_f64() {
        let bytes = 1.0f64.to_be_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f64> = matcher.match_number(Endianness::Big);
        assert_eq!(num, Some(1.0));

        let bytes = 1.0f64.to_le_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f64> = matcher.match_number(Endianness::Little);
        assert_eq!(num, Some(1.0));

        // Too large to fit in f32, should become infinite
        let bytes = (1.0_f64 / 0.0_f64).to_be_bytes();
        let f = f64::from_be_bytes(bytes);
        assert!(f.is_infinite());
        let bytes = f.to_be_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f64> = matcher.match_number(Endianness::Big);
        assert_eq!(num, Some(f));
    }

    #[test]
    fn matcher_test_utf8() {
        let mut buf = [0; 4];
        let codepoints = 'ø'.encode_utf8(&mut buf);
        let mut matcher = Matcher::with_slice(codepoints.as_bytes().into());
        let c: Option<char> = matcher.match_utf8();
        assert_eq!(c, Some('ø'));
    }

    #[test]
    fn matcher_test_utf16() {
        let mut buf = [0; 2];
        let codepoints = 'ø'.encode_utf16(&mut buf);
        let bytes = unsafe {
            let ptr = codepoints.as_ptr() as *const u8;
            let len = codepoints.len() * 2;
            core::slice::from_raw_parts(ptr, len)
        };
        let mut matcher = Matcher::with_slice(bytes.into());
        let c: Option<char> = matcher.match_utf16(Endianness::Native);
        assert_eq!(c, Some('ø'));

        let expected = '😀';
        let mut buf = [0; 2];
        let codepoints = expected.encode_utf16(&mut buf);
        let bytes = unsafe {
            let ptr = codepoints.as_ptr() as *const u8;
            let len = codepoints.len() * 2;
            core::slice::from_raw_parts(ptr, len)
        };
        let mut matcher = Matcher::with_slice(bytes.into());
        let c: Option<char> = matcher.match_utf16(Endianness::Native);
        assert_eq!(c, Some(expected));
    }

    #[test]
    fn matcher_test_utf32() {
        let expected = '😀';
        let codepoints = u32::to_ne_bytes(expected as u32);
        let bytes = codepoints.as_slice();
        let mut matcher = Matcher::with_slice(bytes.into());
        let c: Option<char> = matcher.match_utf32(Endianness::Native);
        assert_eq!(c, Some(expected));
    }

    #[test]
    fn matcher_test_bytes() {
        let buf = b"hello world";
        let mut matcher = Matcher::with_slice(buf.as_slice().into());
        let bytes = matcher.match_bytes(5).unwrap();
        assert!(bytes == b"hello");
    }

    #[test]
    fn matcher_test_bits() {
        let buf = b"hello world";
        let mut matcher = Matcher::with_slice(buf.as_slice().into());
        let bytes = matcher.match_bits(10).unwrap();
        let expected = Selection::new(&[104, 0b01000000], 0, 0, None, 10).unwrap();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn matcher_test_any() {
        let buf = b"hello";
        let mut matcher = Matcher::with_slice(buf.as_slice().into());
        let bytes = matcher.match_bits(10).unwrap();
        let expected = Selection::new(buf.as_slice(), 0, 0, None, 10).unwrap();
        assert_eq!(bytes, expected);
        let rest = matcher.match_any();
        let expected = Selection::new(buf.as_slice(), 1, 2, None, 30).unwrap();
        assert_eq!(rest, expected);
    }
}

use super::*;

/// This struct is used to perform pattern matching against a bitstring/binary.
///
/// It operates on a captured BitSlice so that the internal machinery for traversing
/// bits/bytes can be handled in one place, and to keep this struct focused on the
/// pattern matching semantics.
pub struct Matcher<'a> {
    selection: Selection<'a>,
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
        for byte in self.selection.iter().take(N) {
            unsafe {
                *ptr.add(idx) = byte;
            }
            idx += 1;
        }
        true
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
        N: FromEndianBytes<S> + core::ops::Shr<usize, Output = N>,
    {
        assert!(
            bitsize <= (S * 8),
            "requested bit size is larger than that of the containing type"
        );

        if let Ok(selection) = self.selection.take(bitsize) {
            let mut bytes = [0u8; S];

            let mut matcher = Matcher::new(selection);
            if matcher.read_bytes(&mut bytes) {
                match endianness {
                    Endianness::Native => {
                        let n = N::from_ne_bytes(bytes);
                        if cfg!(target_endian = "big") {
                            // No shift needed, as the bytes are already in big-endian order
                            Some(n)
                        } else {
                            // We need to shift the bytes right so that the least-significant
                            // bits begin at the correct byte boundary
                            let shift = 8 - (bitsize % 8);
                            Some(n >> shift)
                        }
                    }
                    // Big-endian bytes never require a shift
                    Endianness::Big => Some(N::from_be_bytes(bytes)),
                    // Little-endian bytes always require a shift
                    Endianness::Little => {
                        let shift = 8 - (bitsize % 8);
                        let n = N::from_le_bytes(bytes);
                        Some(n >> shift)
                    }
                }
            } else {
                None
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
    pub fn match_utf16(&mut self) -> Option<char> {
        let decoder = Utf16CodepointReader::new(Matcher::new(self.selection));
        let c = char::decode_utf16(decoder)
            .take(1)
            .fold(None, |_acc, c| c.ok())?;
        self.selection = self.selection.shrink_front(char::len_utf16(c) * 16);
        Some(c)
    }

    /// Matches a single utf-32 character from the input
    pub fn match_utf32(&mut self) -> Option<char> {
        let raw: u32 = self.read_number(Endianness::Native)?;
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
    pub fn match_binary(self) -> Option<Selection<'a>> {
        if self.selection.is_binary() {
            Some(self.selection)
        } else {
            None
        }
    }

    /// This operation always succeeds by returning what remains of the underlyings slice
    ///
    /// Since this function consumes the matcher itself, it must always be the last match performed
    pub fn match_any(self) -> Selection<'a> {
        self.selection
    }
}

struct Utf16CodepointReader<'a> {
    matcher: Matcher<'a>,
}
impl<'a> Utf16CodepointReader<'a> {
    fn new(matcher: Matcher<'a>) -> Self {
        Self { matcher }
    }
}
impl<'a> Iterator for Utf16CodepointReader<'a> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.matcher.match_number(Endianness::Native)
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
    fn matcher_test_floats() {
        let bytes = 1.0f64.to_be_bytes();
        let mut matcher = Matcher::with_slice(bytes.as_slice().into());
        let num: Option<f64> = matcher.match_number(Endianness::Big);
        assert_eq!(num, Some(1.0));
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
        let c: Option<char> = matcher.match_utf16();
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
        let c: Option<char> = matcher.match_utf16();
        assert_eq!(c, Some(expected));
    }

    #[test]
    fn matcher_test_utf32() {
        let expected = '😀';
        let codepoints = u32::to_ne_bytes(expected as u32);
        let bytes = codepoints.as_slice();
        let mut matcher = Matcher::with_slice(bytes.into());
        let c: Option<char> = matcher.match_utf32();
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

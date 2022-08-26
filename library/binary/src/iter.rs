use core::iter;

use crate::{Bitstring, MaybePartialByte, Selection};

/// Represents iteration over the bytes in a selection, which may constitute
/// either a binary or bitstring.
///
/// Iteration may produce a trailing partial byte, of which all unused bits will
/// be zeroed.
#[derive(Debug)]
pub struct ByteIter<'a> {
    selection: Selection<'a>,
}
impl<'a> Clone for ByteIter<'a> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            selection: self.selection,
        }
    }
}
impl<'a> ByteIter<'a> {
    pub fn new(selection: Selection<'a>) -> Self {
        Self { selection }
    }

    pub fn from_slice(data: &'a [u8]) -> Self {
        Self {
            selection: Selection::all(data),
        }
    }

    /// In some cases, the underlying bytes can be directly accessed allowing for
    /// more optimal access patterns, see SpecExtend impl for BitVec for an example
    #[inline]
    pub fn as_slice(&self) -> Option<&[u8]> {
        self.selection.as_bytes()
    }
}
impl<'a> Iterator for ByteIter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.selection.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }
}
impl<'a> iter::ExactSizeIterator for ByteIter<'a> {
    fn len(&self) -> usize {
        self.selection.byte_size()
    }

    fn is_empty(&self) -> bool {
        self.selection.byte_size() == 0
    }
}
impl<'a> iter::FusedIterator for ByteIter<'a> {}
unsafe impl<'a> iter::TrustedLen for ByteIter<'a> {}

/// Like `ByteIter`, but intended for cases where special care must be
/// taken around a trailing partial byte, if one is present. This iterator
/// works like `ByteIter` until it encounters a trailing partial byte, in which
/// case it will NOT emit the final byte at all, and instead it must be requested
/// explicitly from the iterator and handled manually.
#[derive(Debug)]
pub struct BitsIter<'a> {
    selection: Selection<'a>,
}
impl<'a> Clone for BitsIter<'a> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            selection: self.selection,
        }
    }
}
impl<'a> BitsIter<'a> {
    pub fn new(selection: Selection<'a>) -> Self {
        Self { selection }
    }

    /// Returns the size in bytes (including trailing partial byte) of the underlying selection
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.selection.byte_size()
    }

    /// Takes the selection from this iterator, consuming it
    ///
    /// This is how users of this iterator must consume the final trailing byte.
    #[inline]
    pub fn consume(self) -> Option<MaybePartialByte> {
        match self.selection {
            Selection::Byte(b) if b.is_partial() => Some(b),
            _ => None,
        }
    }

    /// In some cases, the underlying bytes can be directly accessed allowing for
    /// more optimal access patterns, see SpecExtend impl for BitVec for an example
    #[inline]
    pub fn as_slice(&self) -> Option<&[u8]> {
        self.selection.as_bytes()
    }
}
impl<'a> Iterator for BitsIter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.selection {
            Selection::Empty => None,
            Selection::Byte(b) if b.is_partial() => None,
            _ => self.selection.pop(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }
}
impl<'a> iter::ExactSizeIterator for BitsIter<'a> {
    fn len(&self) -> usize {
        let size = self.selection.byte_size();
        match self.selection.trailing_bits() {
            0 => size,
            _ if size == 0 => 0,
            _ => size - 1,
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl<'a> iter::FusedIterator for BitsIter<'a> {}
unsafe impl<'a> iter::TrustedLen for BitsIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_iter_test_aligned_binary() {
        let bytes = b"hello";

        let selection = Selection::new(bytes.as_slice(), 0, 0, None, 40).unwrap();
        let mut iter = ByteIter::new(selection);

        assert_eq!(iter.next(), Some(104));
        assert_eq!(iter.next(), Some(101));
        assert_eq!(iter.next(), Some(108));
        assert_eq!(iter.next(), Some(108));
        assert_eq!(iter.next(), Some(111));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn byte_iter_test_aligned_bitstring() {
        let bytes = b"hello";

        let selection = Selection::new(bytes.as_slice(), 0, 0, None, 30).unwrap();
        let mut iter = ByteIter::new(selection);

        assert_eq!(iter.next(), Some(104));
        assert_eq!(iter.next(), Some(101));
        assert_eq!(iter.next(), Some(108));
        assert_eq!(iter.next(), Some(0b01101100));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn byte_iter_test_unaligned() {
        let bytes = b"hello";

        let selection = Selection::new(bytes.as_slice(), 0, 1, None, 39).unwrap();
        let mut iter = ByteIter::new(selection);

        assert_eq!(iter.next(), Some(0b11010000));
        assert_eq!(iter.next(), Some(0b11001010));
        assert_eq!(iter.next(), Some(0b11011000));
        assert_eq!(iter.next(), Some(0b11011000));
        assert_eq!(iter.next(), Some(0b11011110));
        assert_eq!(iter.next(), None);
    }
}

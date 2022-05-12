use core::iter;
use core::ops::Range;

/// An iterator over the bytes in a bitstring/binary
///
/// When iterating over the bytes in a sub-binary/bitslice, the bit offset
/// is corrected so that each byte appears as it would if it was aligned to
/// a byte boundary. Extra bits are always discarded (i.e. zeroed).
pub struct ByteIter<'a> {
    data: &'a [u8],
    last: usize,
    offset: u8,
    is_binary: bool,
    num_bits: usize,
    alive: Range<usize>,
}
impl<'a> ByteIter<'a> {
    /// Create a new byte iterator from the given byte slice, bit offset, and length in bits of the data
    /// contained in the byte slice.
    ///
    /// NOTE: This function will panic if the required number of bytes expressed by the offset and length
    /// in bits is greater than the number of bytes present in the slice
    pub fn new(data: &'a [u8], bit_offset: u8, num_bits: usize) -> Self {
        let total_bits = num_bits + bit_offset;
        let len = (total_bits / 8) + ((total_bits % 8 > 0) as usize);
        assert!(data.len() >= len);
        Self {
            data,
            last: len - 1,
            offset: bit_offset,
            is_binary: num_bits % 8 == 0,
            num_bits,
            alive: 0..len,
        }
    }
}
impl<'a> Iterator for ByteIter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        // If no offset is present, we can simply walk the data byte-by-byte
        if self.offset == 0 {
            let idx = self.alive.next()?;
            if self.is_binary {
                return Some(self.data[idx]);
            }

            // If this is not a binary, we need to handle the case where we're at
            // the last byte with extra bits that aren't included in the selection
            let is_last = idx == self.last;
            if is_last {
                // Special-case handling for when the desired number of bits is contained in a single byte
                let is_first = idx == 0;
                if is_first {
                    let num_bits: u8 = self.num_bits.try_into().unwrap();
                    let discard_mask = (-1i8 as u8) << (8 - num_bits);
                    return Some(self.data[idx] & discard_mask);
                }

                // Calculate the number of leftover bits
                let remaining_bits: u8 = (self.num_bits - (idx * 8)).try_into().unwrap();
                let discard_bits = 8 - remaining_bits;
                let discard_mask = (-1i8 as u8) << discard_bits;
                return Some(self.data[idx] & discard_mask);
            }
        }

        // If there is an offset though, each byte is composed of bits in
        // two bytes, and special handling is required for the first and last bytes
        let idx = self.alive.next()?;
        let is_first = idx == 0;
        let is_last = idx == (self.data.len() - 1);
        // The shift required to correct for the bit offset
        let left_offset = self.offset;
        // The shift required to move bits from the next byte right, so they can be bitwise-or'd into the output byte
        let right_offset = 8 - left_offset;
        // The mask for bits from the next byte which need to be pulled into the output byte
        let mask = !(-1i8 as u8 >> left_offset);

        // The first byte only requires offset, and grabbing the remaining bits, if applicable from the next byte
        if is_first {
            let left_bits = self.data[idx] << left_offset;

            // Special-case handling for when the first byte is the last byte, and we
            // just needed to shift the bits
            if is_last {
                // Handle discarded bits
                let remaining_bits: u8 = self.num_bits.try_into().unwrap();
                let discard_bits = 8 - remaining_bits;
                let discard_mask = (-1i8 as u8) << discard_bits;
                return Some(left_bits & discard_mask);
            }

            // Otherwise, grab the missing bits from the next byte, shift them into position,
            // then combine the left and right bits together for the output byte
            let right_bits = (self.data[idx + 1] & mask) >> right_offset;
            return Some(left_bits | right_bits);
        }

        // The last byte only requires shifting the leftover bits
        if is_last {
            // Handle discarded bits
            let remaining_bits: u8 = (self.num_bits - (idx * 8)).try_into().unwrap();
            let discard_bits = 8 - remaining_bits;
            let discard_mask = (-1i8 as u8) << discard_bits;
            let left_bits = (self.data[idx] & !mask) << left_offset;
            return Some(left_bits & discard_mask);
        }

        // All intermediate bytes require us to mask/shift relevant bits from both the current byte and the next byte
        let left_bits = (self.data[idx] & !mask) << left_offset;
        let right_bits = (self.data[idx + 1] & mask) >> right_offset;
        Some(left_bits | right_bits)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}
impl<'a> iter::DoubleEndedIterator for ByteIter<'a> {
    // If we're working back -> front, the algorithm is identical to front -> back, except that our
    // index is managed in a decrementing fashion
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.offset == 0 {
            let idx = self.alive.next_back()?;
            if self.is_binary {
                return Some(self.data[idx]);
            }

            // If this is not a binary, we need to handle the case where we're at
            // the last byte with extra bits that aren't included in the selection
            let is_last = idx == self.last;
            if is_last {
                // Special-case handling for when the desired number of bits is contained in a single byte
                let is_first = idx == 0;
                if is_first {
                    let num_bits: u8 = self.num_bits.try_into().unwrap();
                    let mask = (-1i8 as u8) << (8 - num_bits);
                    return Some(self.data[idx] & mask);
                }

                // Calculate the number of leftover bits
                let remaining_bits: u8 = (self.num_bits - (idx * 8)).try_into().unwrap();
                let discard_bits = 8 - remaining_bits;
                let mask = (-1i8 as u8) << discard_bits;
                return Some(self.data[idx] & mask);
            }
        }

        let idx = self.alive.next_back()?;
        let is_first = idx == 0;
        let is_last = idx == (self.data.len() - 1);
        let left_offset = self.offset;
        let right_offset = 8 - left_offset;
        let mask = !(-1i8 as u8 >> left_offset);

        if is_first {
            let left_bits = self.data[idx] << left_offset;

            if is_last {
                // Handle discarded bits
                let remaining_bits: u8 = self.num_bits.try_into().unwrap();
                let discard_bits = 8 - remaining_bits;
                let discard_mask = (-1i8 as u8) << discard_bits;
                return Some(left_bits & discard_mask);
            }

            let right_bits = (self.data[idx + 1] & mask) >> right_offset;
            return Some(left_bits | right_bits);
        }

        if is_last {
            // Handle discarded bits
            let remaining_bits: u8 = (self.num_bits - (idx * 8)).try_into().unwrap();
            let discard_bits = 8 - remaining_bits;
            let discard_mask = (-1i8 as u8) << discard_bits;
            let left_bits = (self.data[idx] & !mask) << left_offset;
            return Some(left_bits & discard_mask);
        }

        let left_bits = (self.data[idx] & !mask) << left_offset;
        let right_bits = (self.data[idx + 1] & mask) >> right_offset;
        Some(left_bits | right_bits)
    }
}
impl<'a> iter::ExactSizeIterator for ByteIter<'a> {
    fn len(&self) -> usize {
        self.alive.end - self.alive.start
    }

    fn is_empty(&self) -> bool {
        self.alive.is_empty()
    }
}
impl<'a> iter::FusedIterator for ByteIter<'a> {}
impl<'a> iter::TrustedLen for ByteIter<'a> {}

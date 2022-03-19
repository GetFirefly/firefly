use crate::{BitCarrier, BitTransport, BitRead, BitSlice};

/// Extends the inner carrier by the given number of words.
///
/// Even if `head_extend` and `tail_extend` is 0, this will pad the length of
/// inner to the next whole transport size.
#[derive(Debug, Clone)]
pub struct ExtendWords<I: BitCarrier> {
    pub(crate) inner: I,
    pub(crate) extend: I::T,
    pub(crate) head_extend: usize,
    pub(crate) tail_extend: usize,
}

impl<I> BitCarrier for ExtendWords<I>
where
    I: BitCarrier,
{
    type T = I::T;
    fn bit_len(&self) -> usize {
        (((self.inner.bit_len() + 7) / 8) + self.head_extend + self.tail_extend) * I::T::BIT_SIZE
    }
}

impl<I> BitRead for ExtendWords<I>
where
    I: BitRead,
{
    fn read_word(&self, n: usize) -> I::T {
        if n < self.head_extend {
            self.extend
        } else {
            let n = n - self.head_extend;

            let inner_bl = self.inner.bit_len();
            let last_word = inner_bl / 8;

            let inner_bl_mod = inner_bl % I::T::BIT_SIZE;

            if inner_bl == 0 {
                self.extend
            } else if n < last_word || (n == last_word && inner_bl_mod == 0) {
                // We read a whole word, no funny business
                self.inner.read_word(n)
            } else if n == last_word {
                debug_assert!(inner_bl_mod != 0);

                // We read a partial word.
                // We need to extend the last bits.
                let word = self.inner.read_word(n);
                let mod_inv = I::T::BIT_SIZE - inner_bl_mod;
                let mask = I::T::MAX << mod_inv as u32;
                (word & mask) | (self.extend & !mask)
            } else if n > last_word {
                self.extend
            } else {
                unreachable!()
            }
        }
    }
}

pub type Extend<I> = BitSlice<ExtendWords<I>>;

pub fn extend_words_pad<I: BitCarrier>(inner: I, extend: I::T, head_words: usize, tail_words: usize) -> ExtendWords<I> {
    ExtendWords {
        inner,
        extend,
        head_extend: head_words,
        tail_extend: tail_words,
    }
}

pub fn extend_bits<I: BitCarrier>(inner: I, extend: I::T, head_bits: usize, tail_bits: usize) -> Extend<I> {
    let head_words = (head_bits + (I::T::BIT_SIZE - 1)) / I::T::BIT_SIZE;
    let tail_words = (tail_bits + (I::T::BIT_SIZE - 1)) / I::T::BIT_SIZE;

    let offset = (head_words * I::T::BIT_SIZE) - head_bits;
    let length = head_bits + inner.bit_len() + tail_bits;

    inner.extend_words_pad(extend, head_words, tail_words).slice_bits(offset, length)
}

#[cfg(test)]
mod tests {
    use crate::{BitSlice, BitCarrier, BitRead};

    #[test]
    fn extend_words_pad() {
        let bs = BitSlice::with_offset_length(0b10010110u8, 2, 6);

        let bse = (&bs).extend_words_pad(0xff, 0, 0);
        assert_eq!(bse.bit_len(), 8);
        assert_eq!(bse.read_word(0), 0b01011011);

        let bse = (&bs).extend_bits(0xff, 4, 4);
        assert_eq!(bse.bit_len(), 6 + 8);
        assert_eq!(bse.read_word(0), 0b11110101);
        assert_eq!(bse.read_word(1) & 0b11111100, 0b10111100);
    }

}

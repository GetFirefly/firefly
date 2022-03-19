use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use super::BitSlice;
use super::{BitCarrier, BitRead, BitWrite};

#[derive(Debug, Clone)]
pub struct BitVec {
    buf: Vec<u8>,
    bit_size: usize,
}

impl BitVec {
    pub fn new() -> Self {
        BitVec {
            buf: vec![],
            bit_size: 0,
        }
    }

    pub fn with_size(size: usize) -> Self {
        let byte_len = (size + 7) / 8;
        BitVec {
            buf: vec![0; byte_len],
            bit_size: size,
        }
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.bit_size = 0;
    }

    pub fn empty(&mut self) -> bool {
        self.bit_size == 0
    }

    pub fn try_as_byte_aligned_slice(&self) -> Option<&[u8]> {
        if self.bit_size % 8 == 0 {
            Some(&self.buf)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn as_ref(&self) -> &[u8] {
        &self.buf
    }

    pub fn get(&self, n: usize) -> Option<u8> {
        self.buf.get(n).cloned()
    }

    pub fn push<R>(&mut self, from: R)
    where
        R: BitRead<T = u8>,
    {
        let needed = from.bit_len();
        let availible = self.buf.len() * 8 - self.bit_size;

        if needed > availible {
            let needed_bytes = ((needed - availible) + 7) / 8;
            for _ in 0..needed_bytes {
                self.buf.push(0);
            }
        }

        let mut slice =
            BitSlice::<&mut [u8]>::with_offset_length(&mut self.buf[..], self.bit_size, needed);
        slice.write(from);

        self.bit_size += needed;
    }
    pub fn pop<R>(&mut self, to: &mut R) -> Option<()>
    where
        R: BitWrite<T = u8>,
    {
        let to_size = to.bit_len();
        let self_size = self.bit_size;
        if self_size < to_size {
            return None;
        }

        let slice = BitSlice::<&mut [u8]>::with_offset_length(
            &mut self.buf[..],
            self.bit_size - to_size,
            to_size,
        );
        to.write(&slice);

        let unneeded = to_size / 8;
        for _ in 0..unneeded {
            self.buf.pop();
        }

        self.bit_size -= to_size;

        debug_assert!(self.bit_size <= self.buf.len() * 8);

        Some(())
    }

    pub fn iter_bytes(&self) -> BitVecBytesIterator {
        BitVecBytesIterator {
            vec: self,
            elem: 0,
            rem: self.bit_size + 8,
        }
    }
}

impl Default for BitVec {
    fn default() -> Self {
        BitVec::new()
    }
}

impl From<Vec<u8>> for BitVec {
    fn from(buf: Vec<u8>) -> BitVec {
        BitVec {
            bit_size: buf.len() * 8,
            buf,
        }
    }
}

pub struct BitVecBytesIterator<'a> {
    vec: &'a BitVec,
    elem: usize,
    rem: usize,
}
impl<'a> Iterator for BitVecBytesIterator<'a> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        if self.rem < 8 {
            None
        } else {
            self.rem -= 8;
            let mask = !(!0u8)
                .checked_shr(std::cmp::min(8, self.rem) as u32)
                .unwrap_or(0);
            let ret = self.vec.buf[self.elem] & mask;
            self.elem += 1;
            Some(ret)
        }
    }
}

//impl Eq for BitVec {}
//impl PartialEq for BitVec {
//    fn eq(&self, other: &Self) -> bool {
//        if self.bit_size != other.bit_size { return false; }
//        self.iter_bytes().eq(other.iter_bytes())
//    }
//}
//
//impl Ord for BitVec {
//    fn cmp(&self, other: &Self) -> Ordering {
//        match self.iter_bytes().cmp(other.iter_bytes()) {
//            Ordering::Equal => (),
//            non_eq => return non_eq,
//        }
//
//        self.bit_size.cmp(&other.bit_size)
//    }
//}
//impl PartialOrd for BitVec {
//    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//        Some(self.cmp(other))
//    }
//}
//
//impl Hash for BitVec {
//    fn hash<H>(&self, state: &mut H) where H: Hasher {
//        self.bit_size.hash(state);
//        for elem in self.iter_bytes() {
//            elem.hash(state);
//        }
//    }
//}

impl BitCarrier for BitVec {
    type T = u8;
    fn bit_len(&self) -> usize {
        self.bit_size
    }
}
impl BitRead for BitVec {
    fn read_word(&self, n: usize) -> u8 {
        self.buf.get(n).cloned().unwrap_or(0)
    }
}
impl BitWrite for BitVec {
    fn write_word(&mut self, n: usize, data: u8, mask: u8) {
        self.buf
            .get_mut(n)
            .map(|d| *d = (*d & !mask) | (data & mask));
    }
}

impl Eq for BitVec {}
impl<O> PartialEq<O> for BitVec
where
    O: BitRead<T = u8>,
{
    fn eq(&self, other: &O) -> bool {
        if self.bit_len() != other.bit_len() {
            return false;
        }
        self.iter_words().eq(other.iter_words())
    }
}

impl<O> PartialOrd<O> for BitVec
where
    O: BitRead<T = u8>,
{
    fn partial_cmp(&self, other: &O) -> Option<Ordering> {
        match self.iter_words().cmp(other.iter_words()) {
            Ordering::Equal => (),
            non_eq => return Some(non_eq),
        }
        Some(self.bit_len().cmp(&other.bit_len()))
    }
}
impl Ord for BitVec {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Hash for BitVec {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.bit_len().hash(state);
        for elem in self.iter_words() {
            elem.hash(state);
        }
    }
}

#[cfg(test)]
mod test {
    use super::BitVec;

    #[test]
    fn basic_push_pop() {
        let mut vec = BitVec::new();

        vec.push(1u32);
        assert!(vec.len() == 4);
        assert!(vec.get(0) == Some(0));
        assert!(vec.get(1) == Some(0));
        assert!(vec.get(2) == Some(0));
        assert!(vec.get(3) == Some(1));
        assert!(vec.get(4) == None);

        let mut ret: u32 = 0;
        vec.pop(&mut ret);
        assert!(vec.len() == 0);
        assert!(ret == 1);

        vec.clear();
        vec.push(true);
        vec.push(1u32);
        assert!(vec.len() == 5);
        assert!(vec.get(0) == Some(128));
        assert!(vec.get(1) == Some(0));
        assert!(vec.get(2) == Some(0));
        assert!(vec.get(3) == Some(0));
        assert!(vec.get(4) == Some(128));
        assert!(vec.get(5) == None);

        let mut ret: bool = false;
        vec.pop(&mut ret);
        assert!(ret == true);

        let mut ret: u32 = 0;
        vec.pop(&mut ret);
        assert!(ret == 0b10000000_00000000_00000000_00000000);
    }
}

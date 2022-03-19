use super::{BitCarrier, BitRead, BitTransport, BitWrite};

macro_rules! impl_prim {
    ($typ:ty, $utyp:ty) => {
        impl BitCarrier for $typ {
            type T = u8;
            fn bit_len(&self) -> usize {
                std::mem::size_of::<Self>() * 8
            }
        }
        impl BitRead for $typ {
            fn read_word(&self, n: usize) -> u8 {
                let size = std::mem::size_of::<Self>();
                if n < size {
                    ((*self as $utyp) >> (8 * (size - 1 - n))) as u8
                } else {
                    0
                }
            }
        }
        impl BitWrite for $typ {
            fn write_word(&mut self, n: usize, data: u8, mask: u8) {
                let size = std::mem::size_of::<Self>();
                let offset = 8 * (size - 1 - n);
                *self = (((*self as $utyp) & !((mask as $utyp) << offset))
                    | ((data as $utyp) << offset)) as $typ;
            }
        }
    };
}

macro_rules! impl_float {
    ($typ:ty, $utyp:ty) => {
        impl BitCarrier for $typ {
            type T = u8;
            fn bit_len(&self) -> usize {
                std::mem::size_of::<Self>() * 8
            }
        }
        impl BitRead for $typ {
            fn read_word(&self, n: usize) -> u8 {
                let size = std::mem::size_of::<Self>();
                if n < size {
                    ((self.to_bits()) >> (8 * (size - 1 - n))) as u8
                } else {
                    0
                }
            }
        }
        impl BitWrite for $typ {
            fn write_word(&mut self, n: usize, data: u8, mask: u8) {
                let size = std::mem::size_of::<Self>();
                let offset = 8 * (size - 1 - n);
                *self = Self::from_bits(
                    ((self.to_bits()) & !((mask as $utyp) << offset)) | ((data as $utyp) << offset),
                );
            }
        }
    };
}

impl_prim!(u8, u8);
impl_prim!(i8, u8);
impl_prim!(u16, u16);
impl_prim!(i16, u16);
impl_prim!(u32, u32);
impl_prim!(i32, u32);
impl_prim!(u64, u64);
impl_prim!(i64, u64);
impl_prim!(u128, u128);
impl_prim!(i128, u128);

impl_float!(f32, u32);
impl_float!(f64, u64);

// Base u1 implementations
impl BitCarrier for bool {
    type T = u8;
    fn bit_len(&self) -> usize {
        1
    }
}
impl BitRead for bool {
    fn read_word(&self, _n: usize) -> u8 {
        (*self as u8) << 7
    }
}
impl BitWrite for bool {
    fn write_word(&mut self, n: usize, data: u8, mask: u8) {
        assert!(n == 0);
        if mask & 0b10000000 != 1 {
            *self = (data & 0b10000000) != 0;
        } else {
            panic!()
        }
    }
}

// Blanket implementations
impl<N, Tr> BitCarrier for &N
where
    N: BitCarrier<T = Tr>,
    Tr: BitTransport,
{
    type T = Tr;
    fn bit_len(&self) -> usize {
        N::bit_len(self)
    }
}
impl<N> BitRead for &N
where
    N: BitRead,
{
    fn read_word(&self, n: usize) -> Self::T {
        N::read_word(self, n)
    }
}

impl<N, Tr> BitCarrier for &mut N
where
    N: BitCarrier<T = Tr>,
    Tr: BitTransport,
{
    type T = Tr;
    fn bit_len(&self) -> usize {
        N::bit_len(self)
    }
}
impl<N> BitRead for &mut N
where
    N: BitRead,
{
    fn read_word(&self, n: usize) -> Self::T {
        N::read_word(self, n)
    }
}
impl<N> BitWrite for &mut N
where
    N: BitWrite,
{
    fn write_word(&mut self, n: usize, data: Self::T, mask: Self::T) {
        N::write_word(self, n, data, mask)
    }
}

//impl<T, N> BitCarrier<T> for &mut N where N: BitCarrier<T>, T: BitTransport {
//    fn bit_len(&self) -> usize {
//        N::bit_len(self)
//    }
//}
//impl<T, N> BitRead<T> for &mut N where N: BitRead<T>, T: BitTransport {
//    fn read_word(&self, n: usize) -> T {
//        N::read_word(self, n)
//    }
//}

impl<Tr> BitCarrier for Vec<Tr>
where
    Tr: BitTransport,
{
    type T = Tr;
    fn bit_len(&self) -> usize {
        (self as &[Self::T]).bit_len()
    }
}
impl<Tr> BitRead for Vec<Tr>
where
    Tr: BitTransport,
{
    fn read_word(&self, n: usize) -> Self::T {
        (self as &[Self::T]).read_word(n)
    }
}
impl<Tr> BitWrite for Vec<Tr>
where
    Tr: BitTransport,
{
    fn write_word(&mut self, n: usize, data: Self::T, mask: Self::T) {
        (self as &mut [Self::T]).write_word(n, data, mask)
    }
}

impl<Tr> BitCarrier for &[Tr]
where
    Tr: BitTransport,
{
    type T = Tr;
    fn bit_len(&self) -> usize {
        self.len() * 8
    }
}
impl<Tr> BitRead for &[Tr]
where
    Tr: BitTransport,
{
    fn read_word(&self, n: usize) -> Self::T {
        self.get(n).cloned().unwrap_or(Self::T::ZERO)
    }
}

impl<Tr> BitCarrier for &mut [Tr]
where
    Tr: BitTransport,
{
    type T = Tr;
    fn bit_len(&self) -> usize {
        self.len() * 8
    }
}
impl<Tr> BitRead for &mut [Tr]
where
    Tr: BitTransport,
{
    fn read_word(&self, n: usize) -> Self::T {
        self.get(n).cloned().unwrap_or(Self::T::ZERO)
    }
}
impl<Tr> BitWrite for &mut [Tr]
where
    Tr: BitTransport,
{
    fn write_word(&mut self, n: usize, data: Self::T, mask: Self::T) {
        self.get_mut(n).map(|v| *v = (*v & !mask) | (data & mask));
    }
}

impl<Tr> BitCarrier for [Tr]
where
    Tr: BitTransport,
{
    type T = Tr;
    fn bit_len(&self) -> usize {
        self.len() * 8
    }
}
impl<Tr> BitRead for [Tr]
where
    Tr: BitTransport,
{
    fn read_word(&self, n: usize) -> Self::T {
        self.get(n).cloned().unwrap_or(Self::T::ZERO)
    }
}
impl<Tr> BitWrite for [Tr]
where
    Tr: BitTransport,
{
    fn write_word(&mut self, n: usize, data: Self::T, mask: Self::T) {
        self.get_mut(n).map(|v| *v = (*v & !mask) | (data & mask));
    }
}

use crate::Integer;
use liblumen_binary::{integer_to_carrier, BitCarrier, BitRead, BitSlice, Endianness};

impl BitCarrier for Integer {
    type T = u8;
    fn bit_len(&self) -> usize {
        match self {
            Integer::Big(bi) => bi.bit_len(),
            Integer::Small(si) => si.bit_len(),
        }
    }
}

#[derive(Debug)]
pub enum IntegerBits {
    //SmallBE(SmallIntCarrierBE),
    //SmallLE(SmallIntCarrierLE),
    Big(BitSlice<Vec<u8>>),
}

impl BitCarrier for IntegerBits {
    type T = u8;
    fn bit_len(&self) -> usize {
        match self {
            //Self::SmallBE(si) => si.bit_len(),
            //Self::SmallLE(si) => si.bit_len(),
            Self::Big(bi) => bi.bit_len(),
        }
    }
}
impl BitRead for IntegerBits {
    fn read_word(&self, n: usize) -> u8 {
        match self {
            //Self::SmallBE(si) => si.read_word(n),
            //Self::SmallLE(si) => si.read_word(n),
            Self::Big(bi) => bi.read_word(n),
        }
    }
}

impl Integer {
    pub fn encode_bitstring(&self, bits: usize, endian: Endianness) -> IntegerBits {
        match self {
            Integer::Big(bi) => IntegerBits::Big(integer_to_carrier(bi.clone(), bits, endian)),
            Integer::Small(si) => {
                // TODO: Don't convert
                IntegerBits::Big(integer_to_carrier((*si).into(), bits, endian))

                //let mask = if si < &0 {
                //    0xff
                //} else {
                //    0x00
                //};

                //match endian {
                //    Endianness::Big => {
                //        IntegerBits::SmallBE(SmallIntCarrierBE {
                //            data: *si as u64,
                //            bits,
                //            padding: mask,
                //            bytes: (bits + 7) / 8,
                //        })
                //    }
                //    Endianness::Little => {
                //        IntegerBits::SmallLE(SmallIntCarrierLE {
                //            data: *si as u64,
                //            bits,
                //            padding: mask,
                //        })
                //    }
                //}
            }
        }
    }
}

//impl BitCarrier for IntegerBitVec {
//    type T = u8;
//    fn bit_len(&self) -> usize {
//        match self {
//            Self::Small(si) => i.bit_len(),
//            Self::Big(bi) => bi.bit_len(),
//        }
//    }
//}
//
//impl BitRead for IntegerBitVec {
//    fn read_word(&self, n: usize) -> u8 {
//        match self {
//            Self::Small(si) => si.read_word(n),
//            Self::Big(bi) => bi.read_word(n),
//        }
//    }
//}
//
//impl From<Integer> for IntegerBitVec {
//    fn from(int: Integer) -> Self {
//        match int {
//            Integer::Big(bi) => Self::Big(liblumen_binary::integer_to_carrier(bi, )),
//            Integer::Small(si) => Self::Small(si),
//        }
//    }
//}

//#[derive(Debug)]
//pub struct SmallIntCarrierBE {
//    data: u64,
//    bits: usize,
//    padding: u8,
//    bytes: usize,
//}
//impl BitCarrier for SmallIntCarrierBE {
//    type T = u8;
//    fn bit_len(&self) -> usize {
//        self.bits
//    }
//}
//impl BitRead for SmallIntCarrierBE {
//    fn read_word(&self, n: usize) -> u8 {
//        if n >= self.bytes {
//            return self.padding;
//        }
//
//        let inv = self.bytes - n - 1;
//        println!("inv: {:?}", inv);
//
//        if inv >= 8 {
//            self.padding
//        } else {
//            let d1 = self.data.read_word(7 - inv);
//            let d2 = if inv == 7 {
//                self.padding
//            } else {
//                self.data.read_word(n)
//            };
//
//            self.data.read_word(7 - inv)
//        }
//    }
//}
//
//#[derive(Debug)]
//pub struct SmallIntCarrierLE {
//    data: u64,
//    bits: usize,
//    padding: u8,
//}
//impl BitCarrier for SmallIntCarrierLE {
//    type T = u8;
//    fn bit_len(&self) -> usize {
//        self.bits
//    }
//}
//impl BitRead for SmallIntCarrierLE {
//    fn read_word(&self, n: usize) -> u8 {
//        if n >= 8 {
//            self.padding
//        } else {
//            let offset = self.bits % 8;
//            let d = self.data.read_word(7 - n);
//            if offset == 0 {
//                d
//            } else {
//                d << (8 - offset)
//            }
//        }
//    }
//}

// <<20::big-size(12)>> -> <<1, 4::size(4)>>
// <<20::little-size(12)>> -> <<20, 0::size(4)>>
// <<-20::big-size(12)>> -> <<254, 12::size(4)>>
// <<-20::little-size(12)>> -> <<236, 15::size(4)>>

#[cfg(test)]
mod tests {
    use crate::Integer;
    use liblumen_binary::{BitVec, Endianness};

    #[test]
    fn unsigned_big_endian_bigint() {
        let num = 20;

        let bs1 = Integer::Big(num.into()).encode_bitstring(12, Endianness::Big);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let parts = (b1.get(0).unwrap(), b1.get(1).unwrap());
        assert_eq!(parts, (1, 4 << 4));
    }

    #[test]
    fn unsigned_little_endian_bigint() {
        let num = 20;

        let bs1 = Integer::Big(num.into()).encode_bitstring(12, Endianness::Little);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let parts = (b1.get(0).unwrap(), b1.get(1).unwrap());
        assert_eq!(parts, (20, 0 << 4));
    }

    #[test]
    fn signed_big_endian_bigint() {
        let num = -20;

        let bs1 = Integer::Big(num.into()).encode_bitstring(12, Endianness::Big);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let parts = (b1.get(0).unwrap(), b1.get(1).unwrap());
        assert_eq!(parts, (254, 12 << 4));
    }

    #[test]
    fn signed_little_endian_bigint() {
        let num = -20;

        let bs1 = Integer::Big(num.into()).encode_bitstring(12, Endianness::Little);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let parts = (b1.get(0).unwrap(), b1.get(1).unwrap());
        assert_eq!(parts, (236, 15 << 4));
    }

    #[test]
    fn unsigned_big_endian_smallint() {
        let num = 20;

        let bs1 = Integer::Small(num).encode_bitstring(12, Endianness::Big);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let parts = (b1.get(0).unwrap(), b1.get(1).unwrap());
        assert_eq!(parts, (1, 4 << 4));
    }

    #[test]
    fn unsigned_little_endian_smallint() {
        let num = 20;

        let bs1 = Integer::Small(num).encode_bitstring(12, Endianness::Little);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let parts = (b1.get(0).unwrap(), b1.get(1).unwrap());
        assert_eq!(parts, (20, 0 << 4));
    }

    #[test]
    fn signed_big_endian_smallint() {
        let num = -20;

        let bs1 = Integer::Small(num).encode_bitstring(12, Endianness::Big);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let parts = (b1.get(0).unwrap(), b1.get(1).unwrap());
        assert_eq!(parts, (254, 12 << 4));
    }

    #[test]
    fn signed_little_endian_smallint() {
        let num = -20;

        let bs1 = Integer::Small(num).encode_bitstring(12, Endianness::Little);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let parts = (b1.get(0).unwrap(), b1.get(1).unwrap());
        assert_eq!(parts, (236, 15 << 4));
    }

    #[test]
    fn num_to_bitstring() {
        let num = 0b101001101001;

        let bs1 = Integer::Small(num).encode_bitstring(12, Endianness::Little);
        let mut b1 = BitVec::new();
        b1.push(&bs1);

        let bs2 = Integer::Big(num.into()).encode_bitstring(12, Endianness::Little);
        let mut b2 = BitVec::new();
        b2.push(&bs2);

        println!("{:?}", b1);
        println!("{:?}", b2);
        assert_eq!(b1, b2);
    }
}

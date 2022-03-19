use num_bigint::{BigInt, Sign};

use super::{BitCarrier, BitRead, BitSlice, BitWrite, Endianness};

impl BitCarrier for BigInt {
    type T = u8;
    fn bit_len(&self) -> usize {
        let bits = self.bits() as usize;
        if self.sign() == Sign::Minus {
            bits + 1
        } else {
            bits
        }
    }
}

#[cfg(target_endian = "little")]
const NATIVE_ENDIANNESS: Endianness = Endianness::Little;

#[cfg(target_endian = "big")]
const NATIVE_ENDIANNESS: Endianness = Endianness::Big;

pub fn integer_to_carrier(mut int: BigInt, bits: usize, endian: Endianness) -> BitSlice<Vec<u8>> {
    let negative = int.sign() == Sign::Minus;
    if negative {
        int += 1;
    }

    let endian = match endian {
        Endianness::Native => NATIVE_ENDIANNESS,
        e => e,
    };

    let keep_bytes = (bits + 7) / 8;
    let aux_bits = bits % 8;

    let (_sign, mut digits) = match endian {
        Endianness::Big => int.to_bytes_be(),
        Endianness::Little => int.to_bytes_le(),
        Endianness::Native => unreachable!(),
    };

    match endian {
        Endianness::Big => {
            let mut new = Vec::new();

            if keep_bytes > digits.len() {
                new.resize(keep_bytes - digits.len(), 0);
                new.extend(digits.iter());
            } else {
                new.extend(digits.iter().skip(digits.len() - keep_bytes));
            }

            digits = new;
        }
        Endianness::Little => {
            digits.resize(keep_bytes, 0);
        }
        Endianness::Native => unreachable!(),
    }

    if negative {
        for digit in digits.iter_mut() {
            *digit = !*digit;
        }
    }

    match endian {
        Endianness::Big => {
            if aux_bits > 0 {
                digits[0] &= !(!0 << aux_bits);
                BitSlice::with_offset_length(digits, 8 - aux_bits, bits)
            } else {
                BitSlice::with_offset_length(digits, 0, bits)
            }
        }
        Endianness::Little => {
            if aux_bits > 0 {
                digits[keep_bytes - 1] <<= 8 - aux_bits;
            }
            BitSlice::with_offset_length(digits, 0, bits)
        }
        Endianness::Native => unreachable!(),
    }
}

fn carrier_to_buf<C>(carrier: C, signed: bool, endian: Endianness) -> (Vec<u8>, bool)
where
    C: BitRead<T = u8>,
{
    let bit_len = carrier.bit_len();
    let num_bytes = (bit_len + 7) / 8;

    let aux_bits = bit_len % 8;
    let aux_bits_wrap = if aux_bits == 0 { 8 } else { aux_bits };
    let endian = match endian {
        Endianness::Native => NATIVE_ENDIANNESS,
        e => e,
    };

    let offset = match endian {
        Endianness::Big => 8 - aux_bits_wrap,
        Endianness::Little => 0,
        Endianness::Native => unreachable!(),
    };

    let mut buf = vec![0; num_bytes];
    {
        let mut slice = BitSlice::with_offset_length(&mut buf, offset, bit_len);
        slice.write(carrier);
    }

    let mut last = match endian {
        Endianness::Big => buf[0],
        Endianness::Little => buf[num_bytes - 1] >> aux_bits,
        Endianness::Native => unreachable!(),
    };

    // Sign extend
    let mut sign = false;
    if signed {
        sign = last & (1 << (aux_bits_wrap - 1)) != 0;
        if sign {
            last |= !(!0 >> (8 - aux_bits_wrap));
        }
    }

    match endian {
        Endianness::Big => buf[0] = last,
        Endianness::Little => buf[num_bytes - 1] = last,
        Endianness::Native => unreachable!(),
    }

    (buf, sign)
}

pub fn carrier_to_integer<C>(carrier: C, signed: bool, endian: Endianness) -> BigInt
where
    C: BitRead<T = u8>,
{
    let (mut buf, sign) = carrier_to_buf(carrier, signed, endian);

    let endian = match endian {
        Endianness::Native => NATIVE_ENDIANNESS,
        e => e,
    };

    if sign {
        for elem in buf.iter_mut() {
            *elem = !*elem;
        }
        let mut int = match endian {
            Endianness::Big => BigInt::from_bytes_be(Sign::Plus, &buf),
            Endianness::Little => BigInt::from_bytes_le(Sign::Plus, &buf),
            Endianness::Native => unreachable!(),
        };
        int *= -1;
        int -= 1;
        int
    } else {
        match endian {
            Endianness::Big => BigInt::from_bytes_be(Sign::Plus, &buf),
            Endianness::Little => BigInt::from_bytes_le(Sign::Plus, &buf),
            Endianness::Native => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BitSlice, BitWrite};
    use super::{carrier_to_buf, carrier_to_integer, integer_to_carrier, Endianness};
    use num_bigint::BigInt;

    #[test]
    fn integer_adapter_basic() {
        let int = BigInt::from(0b00001111_00010000);

        {
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Little);
            let mut out: [u8; 2] = [0; 2];
            out.write(&conv);
            assert!(out[0] == 0b00010000);
            assert!(out[1] == 0b00001111);
        }

        {
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Big);
            let mut out: [u8; 2] = [0; 2];
            out.write(&conv);
            assert!(out[0] == 0b00001111);
            assert!(out[1] == 0b00010000);
        }
    }

    #[test]
    fn integer_adapter_unaligned() {
        let int = BigInt::from(0b00001111_00010000);

        {
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Little);

            let mut out: [u8; 2] = [0; 2];
            {
                let mut slice = BitSlice::with_offset_length(&mut out as &mut [u8], 0, 12);
                slice.write(conv);
            }

            assert!(out[0] == 0b00010000);
            assert!(out[1] == 0b11110000);
        }

        {
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Big);

            let mut out: [u8; 2] = [0; 2];
            {
                let mut slice = BitSlice::with_offset_length(&mut out as &mut [u8], 0, 12);
                slice.write(conv);
            }

            assert!(out[0] == 0b11110001);
            assert!(out[1] == 0b00000000);
        }
    }

    #[test]
    fn integer_adapter_negative() {
        {
            let int = BigInt::from(-5);
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Big);
            let mut out: i16 = 0;
            out.write(&conv);
            dbg!(out);
            assert!(out == -5);
        }

        {
            let int = BigInt::from(-10000);
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Big);
            let mut out: i16 = 0;
            out.write(&conv);
            dbg!(out);
            assert!(out == -10000);
        }
    }

    #[test]
    fn integer_carrier_to_buf() {
        {
            let int = BigInt::from(5);
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Big);
            let mut buf: [u8; 2] = [0; 2];
            buf.write(&conv);

            let (out, sign) = carrier_to_buf(&buf as &[u8], false, Endianness::Big);
            assert!(sign == false);
            assert!(out[0] == 0);
            assert!(out[1] == 5);

            let (out, sign) = carrier_to_buf(&buf as &[u8], true, Endianness::Big);
            assert!(out[0] == 0);
            assert!(out[1] == 5);
            assert!(sign == false);
        }

        {
            let int = BigInt::from(0b00000101_01010101);
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Big);

            let mut buf: [u8; 2] = [0; 2];
            let mut carrier = BitSlice::with_offset_length(&mut buf as &mut [u8], 0, 12);

            carrier.write(&conv);

            let (out, sign) = carrier_to_buf(&carrier, false, Endianness::Big);
            assert!(out[0] == 0b00000101);
            assert!(out[1] == 0b01010101);
            assert!(sign == false);

            let (out, sign) = carrier_to_buf(&carrier, true, Endianness::Big);
            assert!(out[0] == 0b00000101);
            assert!(out[1] == 0b01010101);
            assert!(sign == false);
        }

        {
            let int = BigInt::from(0b00001010_10101010);
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Big);

            let mut buf: [u8; 2] = [0; 2];
            let mut carrier = BitSlice::with_offset_length(&mut buf as &mut [u8], 0, 12);

            carrier.write(&conv);

            let (out, sign) = carrier_to_buf(&carrier, false, Endianness::Big);
            assert!(out[0] == 0b00001010);
            assert!(out[1] == 0b10101010);
            assert!(sign == false);

            let (out, sign) = carrier_to_buf(&carrier, true, Endianness::Big);
            assert!(out[0] == 0b11111010);
            assert!(out[1] == 0b10101010);
            assert!(sign == true);
        }

        {
            let int = BigInt::from(0b00001010_10101010);
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Little);

            let mut buf: [u8; 2] = [0; 2];
            let mut carrier = BitSlice::with_offset_length(&mut buf as &mut [u8], 0, 12);

            carrier.write(&conv);

            let (out, sign) = carrier_to_buf(&carrier, false, Endianness::Little);
            assert!(out[0] == 0b10101010);
            assert!(out[1] == 0b00001010);
            assert!(sign == false);

            let (out, sign) = carrier_to_buf(&carrier, true, Endianness::Little);
            assert!(out[0] == 0b10101010);
            assert!(out[1] == 0b11111010);
            assert!(sign == true);
        }
    }

    #[test]
    fn integer_round_trip_basic() {
        {
            let int = BigInt::from(5);
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Big);
            let mut buf: [u8; 2] = [0; 2];
            buf.write(&conv);
            let back = carrier_to_integer(&buf as &[u8], true, Endianness::Big);
            assert!(int == back);
        }

        {
            let int = BigInt::from(-5);
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Big);
            let mut buf: [u8; 2] = [0; 2];
            buf.write(&conv);
            let back = carrier_to_integer(&buf as &[u8], true, Endianness::Big);
            assert!(int == back);
        }

        {
            let int = BigInt::from(5);
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Little);
            let mut buf: [u8; 2] = [0; 2];
            buf.write(&conv);
            let back = carrier_to_integer(&buf as &[u8], true, Endianness::Little);
            assert!(int == back);
        }

        {
            let int = BigInt::from(-5);
            let conv = integer_to_carrier(int.clone(), 16, Endianness::Little);
            let mut buf: [u8; 2] = [0; 2];
            buf.write(&conv);
            let back = carrier_to_integer(&buf as &[u8], true, Endianness::Little);
            assert!(int == back);
        }
    }

    #[test]
    fn integer_round_trip_unaligned() {
        {
            let int = BigInt::from(5);
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Big);

            let mut buf: [u8; 2] = [0; 2];
            let mut carrier = BitSlice::with_offset_length(&mut buf as &mut [u8], 0, 12);

            carrier.write(&conv);

            let back = carrier_to_integer(&carrier, true, Endianness::Big);
            assert!(int == back);
        }

        {
            let int = BigInt::from(5);
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Little);

            let mut buf: [u8; 2] = [0; 2];
            let mut carrier = BitSlice::with_offset_length(&mut buf as &mut [u8], 0, 12);

            carrier.write(&conv);

            let back = carrier_to_integer(&carrier, true, Endianness::Little);
            assert!(int == back);
        }

        {
            let int = BigInt::from(-5);
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Big);

            let mut buf: [u8; 2] = [0; 2];
            let mut carrier = BitSlice::with_offset_length(&mut buf as &mut [u8], 0, 12);

            carrier.write(&conv);

            let back = carrier_to_integer(&carrier, true, Endianness::Big);
            assert!(int == back);
        }

        {
            let int = BigInt::from(-5);
            let conv = integer_to_carrier(int.clone(), 12, Endianness::Little);

            let mut buf: [u8; 2] = [0; 2];
            let mut carrier = BitSlice::with_offset_length(&mut buf as &mut [u8], 0, 12);

            carrier.write(&conv);

            let back = carrier_to_integer(&carrier, true, Endianness::Little);
            assert!(int == back);
        }
    }
}

use core::cmp;
use core::fmt;
use core::mem;
use core::ptr;

use num_bigint::Sign;

use crate::erts::term::prelude::*;

use liblumen_core::sys::Endianness;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryPushFlags(usize);
impl BinaryPushFlags {
    const FLAG_ALIGNED: usize = 1; /* Field is guaranteed to be byte-aligned. */
    const FLAG_LITTLE: usize = 2; /* Field is little-endian (otherwise big-endian). */
    const FLAG_SIGNED: usize = 4; /* Field is signed (otherwise unsigned). */
    const FLAG_EXACT: usize = 8; /* Size in bs_init is exact. */
    const FLAG_NATIVE: usize = 16; /* Native endian. */

    /// Converts an `Encoding` to a raw flags bitset
    #[inline]
    pub fn new(signed: bool, endianness: Endianness) -> Self {
        match endianness {
            Endianness::Little if signed => Self(Self::FLAG_LITTLE | Self::FLAG_SIGNED),
            Endianness::Little => Self(Self::FLAG_LITTLE),
            Endianness::Big if signed => Self(Self::FLAG_SIGNED),
            Endianness::Big => Self(0),
            Endianness::Native if signed => Self(Self::FLAG_NATIVE | Self::FLAG_SIGNED),
            Endianness::Native => Self(Self::FLAG_NATIVE),
        }
    }

    pub fn as_endianness(&self) -> Endianness {
        if self.is_little_endian() {
            Endianness::Little
        } else if self.is_native_endian() {
            Endianness::Native
        } else {
            Endianness::Big
        }
    }

    #[inline]
    pub fn set_aligned(self, aligned: bool) -> Self {
        if aligned {
            Self(self.0 | Self::FLAG_ALIGNED)
        } else {
            Self(self.0 & !Self::FLAG_ALIGNED)
        }
    }

    #[inline(always)]
    pub fn is_aligned(&self) -> bool {
        self.0 & Self::FLAG_ALIGNED == Self::FLAG_ALIGNED
    }

    #[inline(always)]
    pub fn is_signed(&self) -> bool {
        self.0 & Self::FLAG_SIGNED == Self::FLAG_SIGNED
    }

    #[inline(always)]
    pub fn is_little_endian(&self) -> bool {
        self.0 & Self::FLAG_LITTLE == Self::FLAG_LITTLE
    }

    #[inline(always)]
    pub fn is_native_endian(&self) -> bool {
        self.0 & Self::FLAG_NATIVE == Self::FLAG_NATIVE
    }

    #[inline(always)]
    pub fn is_exact_size(&self) -> bool {
        self.0 & Self::FLAG_EXACT == Self::FLAG_EXACT
    }
}
impl Default for BinaryPushFlags {
    fn default() -> Self {
        Self::new(false, Endianness::Big)
    }
}
impl fmt::Debug for BinaryPushFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BinaryPushFlags")
            .field("raw", &format_args!("{:b}", self.0))
            .field("endianness", &format_args!("{:?}", self.as_endianness()))
            .field("is_aligned", &self.is_aligned())
            .field("is_exact_size", &self.is_exact_size())
            .finish()
    }
}

#[repr(C)]
pub struct BinaryPushResult {
    pub builder: *mut BinaryBuilder,
    pub success: bool,
}

/*
 * Here is how many bits we can copy in each reduction.
 *
 * At the time of writing of this comment, CONTEXT_REDS was 4000 and
 * BITS_PER_REDUCTION was 1 KiB (8192 bits). The time for copying an
 * unaligned 4000 KiB binary on my computer (which has a 4,2 GHz Intel
 * i7 CPU) was about 5 ms. The time was approximately 4 times lower if
 * the source and destinations binaries were aligned.
 */

#[allow(unused)]
const BITS_PER_REDUCTION: usize = 8 * 1024;

pub struct BinaryBuilder {
    buffer: Vec<u8>,
    offset: usize,
}
impl BinaryBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            offset: 0,
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.buffer
    }
}

macro_rules! nbytes {
    ($n:expr) => {
        (($n + 7) >> 3)
    };
}

macro_rules! byte_offset {
    ($n:expr) => {
        ($n >> 3)
    };
}

macro_rules! bit_offset {
    ($n:expr) => {
        ($n & 7)
    };
}

impl BinaryBuilder {
    pub fn push_integer(
        &mut self,
        value: Integer,
        num_bits: usize,
        flags: BinaryPushFlags,
    ) -> Result<(), ()> {
        if num_bits == 0 {
            return Ok(());
        }

        let mut bin_offset = self.offset;
        let bit_offset = bit_offset!(bin_offset);
        let mut b;
        let num_bytes = nbytes!(num_bits);

        self.ensure_needed(num_bytes);

        let mut iptr: *mut u8 = self.buffer.as_mut_ptr();

        match value {
            Integer::Small(small) if bit_offset + num_bits <= 8 => {
                let val: isize = small.into();
                let rbits = 8 - bit_offset;
                // All bits are in the same byte
                unsafe {
                    iptr = iptr.offset(byte_offset!(bin_offset) as isize);
                    b = *iptr as usize & (0xFF << rbits);
                    b |= (val as usize & ((1 << num_bits) - 1)) << (8 - bit_offset - num_bits);
                    *iptr = b as u8;
                }
            }
            Integer::Small(_) if bit_offset == 0 => {
                // More than one bit, starting at a byte boundary.
                let offset = fmt_int(
                    unsafe { iptr.offset(byte_offset!(bin_offset) as isize) },
                    num_bytes,
                    value,
                    num_bits,
                    flags,
                )?;
                bin_offset += offset;
            }
            Integer::Small(_) if flags.is_little_endian() => {
                // Small unaligned, little-endian number
                //
                // We must format the number into a temporary buffer, and
                // then copy that into the binary.
                let mut tmp_buffer: Vec<u8> = Vec::with_capacity(num_bytes);
                let tmp_ptr = tmp_buffer.as_mut_ptr();
                let offset = fmt_int(tmp_ptr, num_bytes, value, num_bits, flags)?;
                unsafe {
                    copy_bits(
                        tmp_ptr,
                        0,
                        CopyDirection::Forward,
                        iptr,
                        bin_offset,
                        CopyDirection::Forward,
                        num_bits,
                    );
                }
                bin_offset += offset;
            }
            Integer::Small(small) => {
                let rbits = 8 - bit_offset;

                // Big-endian, more than one byte, but not aligned on a byte boundary.
                // Handle the bits up to the next byte boundary specially, then let
                // fmt_int handle the rest.
                let shift_count = num_bits - rbits;
                let val: isize = small.into();
                unsafe {
                    iptr = iptr.offset(byte_offset!(bin_offset) as isize);
                    b = *iptr as usize & (0xFF << rbits);
                }

                // Shifting with a shift count greater than or equal to the word
                // size may be a no-op (instead of 0 the result may be the unshifted value).
                // Therefore, only do the shift and the OR if the shift count is less than the
                // word size if the number is positive; if negative, we must simulate the sign
                // extension.
                if shift_count < mem::size_of::<usize>() * 8 {
                    b |= (val as usize >> shift_count) & ((1 << rbits) - 1);
                } else if val < 0 {
                    // Simulate sign extension
                    b |= (-1isize & (1 << (rbits as isize) - 1)) as usize;
                }
                unsafe {
                    iptr = iptr.offset(1);
                    *iptr = b as u8;
                }

                // NOTE: fmt_int is known not to fail here
                let offset = fmt_int(
                    iptr,
                    nbytes!(num_bits - rbits),
                    value,
                    num_bits - rbits,
                    flags,
                )?;
                bin_offset += offset;
            }
            Integer::Big(_) if bit_offset == 0 => {
                // Big integer, aligned on a byte boundary.
                // We can format the integer directly into the binary.
                let offset = fmt_int(
                    unsafe { iptr.offset(byte_offset!(bin_offset) as isize) },
                    num_bytes,
                    value,
                    num_bits,
                    flags,
                )?;
                bin_offset += offset;
            }
            Integer::Big(_) => {
                // We must format the number into a temporary buffer, and
                // then copy that into the binary.
                let mut tmp_buffer: Vec<u8> = Vec::with_capacity(num_bytes);
                let tmp_ptr = tmp_buffer.as_mut_ptr();
                let offset = fmt_int(tmp_ptr, num_bytes, value, num_bits, flags)?;
                unsafe {
                    copy_bits(
                        tmp_ptr,
                        0,
                        CopyDirection::Forward,
                        iptr,
                        bin_offset,
                        CopyDirection::Forward,
                        num_bits,
                    );
                }
                bin_offset += offset;
            }
        }

        self.offset = bin_offset + num_bits;

        Ok(())
    }

    // TODO: erts_new_bs_put_float(Process *c_p, Eterm arg, Uint num_bits, int flags);
    pub fn push_float(
        &mut self,
        _value: f64,
        _num_bits: usize,
        _flags: BinaryPushFlags,
    ) -> Result<(), ()> {
        unimplemented!()
    }

    pub fn push_utf8(&mut self, value: isize) -> Result<(), ()> {
        let bin_offset = self.offset;
        let bit_offset;
        let num_bits;
        let mut tmp_buf = [0u8; 4];
        let mut use_tmp = false;
        let mut dst: *mut u8;

        if value < 0 {
            return Err(());
        }

        bit_offset = bit_offset!(bin_offset);
        if bit_offset == 0 {
            // We can write directly into the destination binary
            dst = unsafe {
                self.buffer
                    .as_mut_ptr()
                    .offset(byte_offset!(bin_offset) as isize)
            };
        } else {
            // Unaligned destination binary. Must use a temporary buffer.
            dst = tmp_buf.as_mut_ptr();
            use_tmp = true;
        }

        if value < 0x80 {
            unsafe {
                *dst = value as u8;
            }
            num_bits = 8;
            self.ensure_needed(nbytes!(num_bits));
            if !use_tmp {
                dst = unsafe {
                    self.buffer
                        .as_mut_ptr()
                        .offset(byte_offset!(bin_offset) as isize)
                };
            }
        } else if value < 0x800 {
            num_bits = 16;
            self.ensure_needed(nbytes!(num_bits));
            if !use_tmp {
                dst = unsafe {
                    self.buffer
                        .as_mut_ptr()
                        .offset(byte_offset!(bin_offset) as isize)
                };
            }
            unsafe {
                *dst = 0xC0 | ((value >> 6) as u8);
                *(dst.offset(1)) = 0x80 | ((value & 0x3F) as u8);
            }
        } else if value < 0x10000 {
            if 0xD800 <= value && value <= 0xDFFF {
                return Err(());
            }
            num_bits = 24;
            self.ensure_needed(nbytes!(num_bits));
            if !use_tmp {
                dst = unsafe {
                    self.buffer
                        .as_mut_ptr()
                        .offset(byte_offset!(bin_offset) as isize)
                };
            }
            unsafe {
                *dst = 0xE0 | ((value >> 12) as u8);
                *(dst.offset(1)) = 0x80 | ((value >> 6) as u8 & 0x3F);
                *(dst.offset(2)) = 0x80 | ((value & 0x3F) as u8);
            }
        } else if value < 0x110000 {
            num_bits = 32;
            self.ensure_needed(nbytes!(num_bits));
            if !use_tmp {
                dst = unsafe {
                    self.buffer
                        .as_mut_ptr()
                        .offset(byte_offset!(bin_offset) as isize)
                };
            }
            unsafe {
                *dst = 0xF0 | ((value >> 18) as u8);
                *(dst.offset(1)) = 0x80 | ((value >> 12) as u8 & 0x3F);
                *(dst.offset(2)) = 0x80 | ((value >> 6) as u8 & 0x3F);
                *(dst.offset(3)) = 0x80 | ((value & 0x3F) as u8);
            }
        } else {
            return Err(());
        }

        if bin_offset != 0 {
            unsafe {
                copy_bits(
                    dst,
                    0,
                    CopyDirection::Forward,
                    self.buffer.as_mut_ptr(),
                    bin_offset,
                    CopyDirection::Forward,
                    num_bits,
                );
            }
        }

        self.offset += num_bits;

        Ok(())
    }

    pub fn push_utf16(&mut self, value: isize, flags: BinaryPushFlags) -> Result<(), ()> {
        let bin_offset = self.offset;
        let bit_offset;
        let num_bits;
        let mut tmp_buf = [0u8; 4];
        let mut use_tmp = false;
        let mut dst: *mut u8;

        if value > 0x10FFFF || (0xD800 <= value && value <= 0xDFFF) {
            return Err(());
        }

        bit_offset = bit_offset!(bin_offset);
        if bit_offset == 0 {
            // We can write directly into the destination binary
            dst = unsafe {
                self.buffer
                    .as_mut_ptr()
                    .offset(byte_offset!(bin_offset) as isize)
            };
        } else {
            // Unaligned destination binary. Must use a temporary buffer.
            dst = tmp_buf.as_mut_ptr();
            use_tmp = true;
        }

        if value < 0x10000 {
            num_bits = 16;
            self.ensure_needed(nbytes!(num_bits));
            if !use_tmp {
                dst = unsafe {
                    self.buffer
                        .as_mut_ptr()
                        .offset(byte_offset!(bin_offset) as isize)
                };
            }
            if flags.is_little_endian() {
                unsafe {
                    *dst = value as u8;
                    *(dst.offset(1)) = (value >> 8) as u8;
                }
            } else {
                unsafe {
                    *dst = (value >> 8) as u8;
                    *(dst.offset(1)) = value as u8;
                }
            }
        } else {
            let w1;
            let w2;

            num_bits = 32;
            self.ensure_needed(nbytes!(num_bits));
            if !use_tmp {
                dst = unsafe {
                    self.buffer
                        .as_mut_ptr()
                        .offset(byte_offset!(bin_offset) as isize)
                };
            }
            let value = value - 0x10000;
            let dst = dst as *mut u16;
            w1 = 0xD800 | ((value >> 10) as u16);
            w2 = 0xDC00 | ((value & 0x3FF) as u16);
            if flags.is_little_endian() {
                let w1 = w1.to_le();
                let w2 = w2.to_le();
                unsafe {
                    *dst = w1;
                    *(dst.offset(1)) = w2;
                }
            } else {
                unsafe {
                    *dst = w1;
                    *(dst.offset(1)) = w2;
                }
            }
        }

        if bin_offset != 0 {
            unsafe {
                copy_bits(
                    dst,
                    0,
                    CopyDirection::Forward,
                    self.buffer.as_mut_ptr(),
                    bin_offset,
                    CopyDirection::Forward,
                    num_bits,
                );
            }
        }

        self.offset += num_bits;

        Ok(())
    }

    pub fn push_byte_unit(&mut self, value: Term, unit: u8) -> Result<(), ()> {
        match value.decode().unwrap() {
            TypedTerm::SmallInteger(small_integer) => {
                let bytes = small_integer.to_le_bytes();
                let unit_usize = unit as usize;
                assert!(1 <= unit && unit_usize <= bytes.len());
                self.push_string(&bytes[0..unit_usize])
            }
            _ => unimplemented!("pushing value ({}) as byte with unit ({})", value, unit),
        }
    }

    pub fn push_string(&mut self, value: &[u8]) -> Result<(), ()> {
        self.ensure_needed(value.len());
        let offset = unsafe { write_bytes(self.buffer.as_mut_ptr(), self.offset, value) };
        self.offset += offset;

        Ok(())
    }

    #[inline]
    fn ensure_needed(&mut self, need: usize) {
        self.buffer.resize(need + self.buffer.len(), 0);
    }
}

fn fmt_int(
    buf: *mut u8,
    size_bytes: usize,
    value: Integer,
    size_bits: usize,
    flags: BinaryPushFlags,
) -> Result<usize, ()> {
    let is_signed = flags.is_signed();
    let is_little = flags.is_little_endian();
    match value {
        Integer::Small(small) => {
            assert_ne!(size_bits, 0);

            let v: isize = small.into();
            let bytes = if is_little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            };
            let sign = if is_signed && v < 0 { 1 } else { 0 };
            unsafe { fmt_int_bytes(&bytes, sign, buf, size_bytes, size_bits, flags) }
        }
        Integer::Big(big) => {
            if size_bits == 0 {
                return Ok(0);
            }

            let bytes = if is_little {
                big.to_signed_bytes_le()
            } else {
                big.to_signed_bytes_be()
            };
            let sign = if is_signed && big.sign() == Sign::Minus {
                1
            } else {
                0
            };
            unsafe { fmt_int_bytes(bytes.as_slice(), sign, buf, size_bytes, size_bits, flags) }
        }
    }
}

unsafe fn fmt_int_bytes(
    bytes: &[u8],
    sign: u8,
    buf: *mut u8,
    size_bytes: usize,
    size_bits: usize,
    flags: BinaryPushFlags,
) -> Result<usize, ()> {
    let mut bit_offset = bit_offset!(size_bits);
    let num_bytes = cmp::min(size_bytes, bytes.len());
    if num_bytes < size_bytes {
        let diff = size_bytes - num_bytes;
        // A larger value was requested than what was provided,
        // so sign extend the value accordingly
        if flags.is_little_endian() {
            // Pad right
            let offset = write_bytes(buf, bit_offset, &bytes[0..num_bytes]);
            Ok(pad_bytes(buf, offset, diff, sign))
        } else {
            // Pad left
            bit_offset += pad_bytes(buf, bit_offset, diff, sign);
            Ok(write_bytes(buf, bit_offset, &bytes[0..num_bytes]))
        }
    } else {
        Ok(write_bytes(buf, bit_offset, &bytes[0..num_bytes]))
    }
}

unsafe fn pad_bytes(dst: *mut u8, offset: usize, padding: usize, sign: u8) -> usize {
    use super::primitives::{make_bitmask, mask_bits};

    if offset % 8 == 0 {
        // Byte-aligned
        dst.offset(byte_offset!(offset) as isize)
            .write_bytes(sign, padding);
    } else {
        // Not byte-aligned
        let lmask = make_bitmask(8 - offset as u8);
        // Handle bits in first (unaligned) byte
        let base = dst.offset(byte_offset!(offset) as isize);
        base.write(mask_bits(sign, *base, lmask));
        // All that's left to copy are the remaining bytes, if any
        let padding = padding - 1;
        if padding > 0 {
            base.offset(1).write_bytes(sign, padding);
        }
    }

    offset + (padding * 8)
}

unsafe fn write_bytes(dst: *mut u8, offset: usize, value: &[u8]) -> usize {
    let ptr = value.as_ptr();
    let num_bytes = value.len();
    if bit_offset!(offset) != 0 {
        copy_bits(
            ptr,
            0,
            CopyDirection::Forward,
            dst,
            offset,
            CopyDirection::Forward,
            num_bytes * 8,
        );
    } else {
        let byte_offs = byte_offset!(offset) as isize;
        ptr::copy_nonoverlapping(ptr, dst.offset(byte_offs), num_bytes);
    }

    num_bytes * 8
}

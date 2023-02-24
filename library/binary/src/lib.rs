#![no_std]
#![feature(allocator_api)]
#![feature(test)]
#![feature(trusted_len)]
#![feature(exact_size_is_empty)]
#![feature(str_internals)]
#![feature(const_option_ext)]
#![feature(slice_take)]
#![feature(min_specialization)]
#![feature(extend_one)]
#![cfg_attr(feature = "std", feature(can_vector))]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;
#[cfg(test)]
extern crate test;

use core::fmt;

mod bitvec;
mod flags;
pub mod helpers;
mod iter;
mod matcher;
mod select;
mod spec;
mod traits;

pub use self::bitvec::BitVec;
pub use self::flags::{BinaryFlags, Encoding};
pub use self::iter::{BitsIter, ByteIter};
pub use self::matcher::Matcher;
pub use self::select::{MaybePartialByte, Selection};
pub use self::spec::BinaryEntrySpecifier;
pub use self::traits::{Aligned, Binary, Bitstring, FromEndianBytes, ToEndianBytes};

/// Represents how bytes of a value are laid out in memory:
///
/// Big-endian systems store the most-significant byte at the lowest memory address, and the least-significant
/// byte at the highest memory address.
///
/// Little-endian systems store the least-significant byte at the lowest memory address, and the most-significant
/// byte at the highest memory address.
///
/// When thinking about values like memory addresses or integers, we tend to think about the textual representation,
/// as this is most often what we are presented with when printing them or viewing them in a debugger, and it can be
/// a useful mental model too, however it can be a bit confusing to read them and reason about endianness.
///
/// This is because, generally, when we visualize memory the following rules are used:
///
/// * Bytes of memory are printed left-to-right, i.e. the left is the lowest memory address, and increases as you read towards the right.
/// * Bits of a byte are the opposite; the most-significant bits appear first, decreasing to the least-significant.
///
/// So lets apply that to an example, a 16-bit integer 64542, as viewed on a little-endian machine:
///
/// * 0xfc1e (little-endian hex)
/// * 0x1efc (big-endian hex)
/// * 0b1111110000011110 (little-endian binary)
/// * 0b0001111011111100 (big-endian binary)
///
/// Well that's confusing, The bytes appear to be backwards! The little-endian version has the most-significant bits in the least-significant
/// byte, and the big-endian version has the least-significant bits in the most-significant byte. What's going on here?
///
/// What I find helps with this is to use the following rules instead:
///
/// * Define `origin` as the right-most byte in the sequence
/// * Read bytes starting from the origin, i.e. right-to-left
/// * Read bits within a byte left-to-right as normal (i.e. most-significant bit is on the left)
/// * Endianness determines how to read the bytes from the origin; big-endian has the most-significant byte at the origin,
/// and little-endian has the least-significant byte at the origin
///
/// If we apply those rules to the previous examples, we can see that the textual representation makes more sense now.
/// The little-endian hex is read starting with the least-significant byte 0x1e, followed by 0xfc; while the big-endian
/// integer is read starting with the most-significant byte 0xfc, followed by 0x1e.
///
/// But this can make calculating the value from the text representation a bit awkward still, is there a trick for that?
/// The answer is to normalize out the endianness, so that we can always read a value from most-significant bytes (and bits)
/// left-to-right. When the native endianness matches the endianness of the value (i.e. little-endian value on little-endian
/// machine, or big-endian value on big-endian machine), this is already the case. When reading a value with non-native endianness
/// though, we need to swap the order of the bytes first.
///
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Endianness {
    /// Most-significant bits "first"
    Big = 0,
    Little,
    Native,
}
impl TryFrom<u8> for Endianness {
    type Error = ();

    fn try_from(n: u8) -> Result<Self, Self::Error> {
        match n {
            0 => Ok(Self::Big),
            1 => Ok(Self::Little),
            2 => Ok(Self::Native),
            _ => Err(()),
        }
    }
}
impl fmt::Display for Endianness {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Big => f.write_str("big"),
            Self::Little => f.write_str("little"),
            Self::Native => f.write_str("native"),
        }
    }
}

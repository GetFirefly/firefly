use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};

use crate::erts::term::binary::aligned_binary;
use crate::erts::term::binary::match_context::MatchContext;
use crate::erts::term::binary::sub::SubBinary;
use crate::erts::term::binary::IterableBitstring;

pub trait MaybeAlignedMaybeBinary {
    type Iter: Iterator<Item = u8>;

    unsafe fn as_bytes(&self) -> &[u8];

    fn is_aligned(&self) -> bool;

    fn is_binary(&self) -> bool;

    fn partial_byte_bit_iter(&self) -> Self::Iter;
}

macro_rules! display {
    ($t:ty) => {
        impl Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                if self.is_binary() {
                    if self.is_aligned() {
                        aligned_binary::display(unsafe { self.as_bytes() }, f)
                    } else {
                        let bytes: Vec<u8> = self.full_byte_iter().collect();

                        aligned_binary::display(&bytes, f)
                    }
                } else {
                    f.write_str("<<")?;

                    let mut full_byte_iter = self.full_byte_iter();

                    let has_full_bytes = if let Some(byte) = full_byte_iter.next() {
                        write!(f, "{:#04x?}", byte)?;

                        for byte in full_byte_iter {
                            write!(f, ", {:04x}", byte)?;
                        }

                        true
                    } else {
                        false
                    };

                    let mut partial_byte_bit_iter = self.partial_byte_bit_iter();

                    if let Some(bit) = partial_byte_bit_iter.next() {
                        if has_full_bytes {
                            f.write_str(", ")?;
                        }

                        write!(f, "{} :: 1", bit)?;

                        for bit in partial_byte_bit_iter {
                            write!(f, ", {} :: 1", bit)?;
                        }
                    }

                    f.write_str(">>")
                }
            }
        }
    };
}

display!(MatchContext);
display!(SubBinary);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! hash {
    ($t:ty) => {
        impl Hash for $t {
            fn hash<H: Hasher>(&self, state: &mut H) {
                for byte in self.full_byte_iter() {
                    byte.hash(state);
                }

                for bit in self.partial_byte_bit_iter() {
                    bit.hash(state);
                }
            }
        }
    };
}

hash!(MatchContext);
hash!(SubBinary);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_eq {
    ($o:tt for $s:ty) => {
        impl PartialEq<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn eq(&self, other: &$o) -> bool {
                if self.is_binary() && other.is_binary() {
                    if self.is_aligned() && other.is_aligned() {
                        unsafe { self.as_bytes().eq(other.as_bytes()) }
                    } else {
                        self.full_byte_iter().eq(other.full_byte_iter())
                    }
                } else {
                    let bytes_equal = if self.is_aligned() && other.is_aligned() {
                        unsafe { self.as_bytes().eq(other.as_bytes()) }
                    } else {
                        self.full_byte_iter().eq(other.full_byte_iter())
                    };

                    bytes_equal || {
                        self.partial_byte_bit_iter()
                            .eq(other.partial_byte_bit_iter())
                    }
                }
            }
        }
    };
}

partial_eq!(SubBinary for SubBinary);
partial_eq!(MatchContext for SubBinary);
partial_eq!(MatchContext for MatchContext);
// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! ord {
    ($t:ty) => {
        impl Ord for $t {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                if self.is_binary() && other.is_binary() {
                    if self.is_aligned() && other.is_aligned() {
                        unsafe { self.as_bytes().cmp(other.as_bytes()) }
                    } else {
                        self.full_byte_iter().cmp(other.full_byte_iter())
                    }
                } else {
                    let bytes_ordering = if self.is_aligned() && other.is_aligned() {
                        unsafe { self.as_bytes().cmp(other.as_bytes()) }
                    } else {
                        self.full_byte_iter().cmp(other.full_byte_iter())
                    };

                    bytes_ordering.then_with(|| {
                        self.partial_byte_bit_iter()
                            .cmp(other.partial_byte_bit_iter())
                    })
                }
            }
        }

        impl PartialOrd for $t {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
    };
}

ord!(SubBinary);
ord!(MatchContext);

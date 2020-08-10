use core::convert::TryInto;
use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};
use core::str;

use alloc::string::String;
use alloc::vec::Vec;

use anyhow::*;
use thiserror::Error;

use crate::erts::term::prelude::Boxed;

use super::aligned_binary;
use super::prelude::{MatchContext, SubBinary};

pub trait MaybeAlignedMaybeBinary {
    /// This function will
    unsafe fn as_bytes_unchecked(&self) -> &[u8];

    fn is_aligned(&self) -> bool;

    fn is_binary(&self) -> bool;
}

macro_rules! display {
    ($t:ty) => {
        impl Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                if self.is_binary() {
                    if self.is_aligned() {
                        aligned_binary::display(unsafe { self.as_bytes_unchecked() }, f)
                    } else {
                        let bytes: Vec<u8> = self.full_byte_iter().collect();

                        aligned_binary::display(&bytes, f)
                    }
                } else {
                    f.write_str("<<")?;

                    let mut full_byte_iter = self.full_byte_iter();

                    let has_full_bytes = if let Some(byte) = full_byte_iter.next() {
                        write!(f, "{}", byte)?;

                        for byte in full_byte_iter {
                            write!(f, ",{}", byte)?;
                        }

                        true
                    } else {
                        false
                    };

                    let mut partial_byte_bit_iter = self.partial_byte_bit_iter();

                    if let Some(bit) = partial_byte_bit_iter.next() {
                        if has_full_bytes {
                            f.write_str(",")?;
                        }

                        let mut partial_byte = bit;
                        let mut partial_byte_bit_len = 1;

                        for bit in partial_byte_bit_iter {
                            partial_byte_bit_len += 1;
                            partial_byte = (partial_byte << 1) | bit;
                        }

                        write!(f, "{}:{}", partial_byte, partial_byte_bit_len)?;
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
                        unsafe { self.as_bytes_unchecked().eq(other.as_bytes_unchecked()) }
                    } else {
                        self.full_byte_iter().eq(other.full_byte_iter())
                    }
                } else {
                    let bytes_equal = if self.is_aligned() && other.is_aligned() {
                        unsafe { self.as_bytes_unchecked().eq(other.as_bytes_unchecked()) }
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
        impl PartialEq<Boxed<$o>> for $s {
            #[inline]
            fn eq(&self, other: &Boxed<$o>) -> bool {
                self.eq(other.as_ref())
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
                        unsafe { self.as_bytes_unchecked().cmp(other.as_bytes_unchecked()) }
                    } else {
                        self.full_byte_iter().cmp(other.full_byte_iter())
                    }
                } else {
                    let bytes_ordering = if self.is_aligned() && other.is_aligned() {
                        unsafe { self.as_bytes_unchecked().cmp(other.as_bytes_unchecked()) }
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

macro_rules! impl_maybe_aligned_try_into {
    ($t:ty) => {
        impl TryInto<String> for $t {
            type Error = anyhow::Error;

            #[inline]
            fn try_into(self) -> Result<String, Self::Error> {
                (&self).try_into()
            }
        }

        impl TryInto<String> for &$t {
            type Error = anyhow::Error;

            fn try_into(self) -> Result<String, Self::Error> {
                if self.is_binary() {
                    if self.is_aligned() {
                        match str::from_utf8(unsafe { self.as_bytes_unchecked() }) {
                            Ok(s) => Ok(s.to_owned()),
                            Err(utf8_error) => Err(utf8_error.into()),
                        }
                    } else {
                        let byte_vec: Vec<u8> = self.full_byte_iter().collect();

                        String::from_utf8(byte_vec)
                            .map_err(|from_utf8_error| from_utf8_error.into())
                    }
                } else {
                    Err(NotABinary.into())
                }
            }
        }

        impl TryInto<String> for Boxed<$t> {
            type Error = anyhow::Error;

            #[inline]
            fn try_into(self) -> Result<String, Self::Error> {
                self.as_ref().try_into()
            }
        }

        impl TryInto<Vec<u8>> for $t {
            type Error = anyhow::Error;

            #[inline]
            fn try_into(self) -> Result<Vec<u8>, Self::Error> {
                (&self).try_into()
            }
        }

        impl TryInto<Vec<u8>> for &$t {
            type Error = anyhow::Error;

            #[inline]
            fn try_into(self) -> Result<Vec<u8>, Self::Error> {
                if self.is_binary() {
                    if self.is_aligned() {
                        Ok(unsafe { self.as_bytes_unchecked().to_vec() })
                    } else {
                        Ok(self.full_byte_iter().collect())
                    }
                } else {
                    Err(NotABinary)
                        .context(format!("{} is a bitstring, but not a binary", self))
                        .map_err(From::from)
                }
            }
        }

        impl TryInto<Vec<u8>> for Boxed<$t> {
            type Error = anyhow::Error;

            #[inline]
            fn try_into(self) -> Result<Vec<u8>, Self::Error> {
                self.as_ref().try_into()
            }
        }
    };
}

impl_maybe_aligned_try_into!(MatchContext);
impl_maybe_aligned_try_into!(SubBinary);

#[derive(Debug, Error)]
#[error("bitstring is not a binary")]
pub struct NotABinary;

use core::fmt;
use core::hash;
use core::str;
use core::convert::TryInto;

use crate::erts::exception::runtime;
use crate::erts::term::prelude::Boxed;

use super::prelude::{HeapBin, ProcBin, BinaryLiteral, Binary, MaybePartialByte, IndexByte};

/// A `BitString` that is guaranteed to always be a binary of aligned bytes
pub trait AlignedBinary: Binary {
    /// Returns the underlying binary data as a byte slice
    fn as_bytes(&self) -> &[u8];

    /// Converts this binary to a `&str` slice.
    ///
    /// This conversion does not move the string, it can be considered as
    /// creating a new reference with a lifetime attached to that of `self`.
    #[inline]
    fn as_str<'a>(&'a self) -> &'a str {
        assert!(
            self.is_latin1() || self.is_utf8(),
            "cannot convert a binary containing non-UTF-8/non-ASCII characters to &str"
        );
        unsafe {
            str::from_utf8_unchecked(self.as_bytes())
        }
    }
}

impl<T: AlignedBinary> IndexByte for T {
    default fn byte(&self, index: usize) -> u8 {
        self.as_bytes()[index]
    }
}

impl<A: AlignedBinary> MaybePartialByte for A {
    #[inline]
    default fn partial_byte_bit_len(&self) -> u8 {
        0
    }

    #[inline]
    default fn total_bit_len(&self) -> usize {
        self.full_byte_len() * 8
    }

    #[inline]
    default fn total_byte_len(&self) -> usize {
        self.full_byte_len()
    }
}

macro_rules! impl_aligned_binary {
    ($t:ty) => {

        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                display(self.as_bytes(), f)
            }
        }

        impl hash::Hash for $t {
            fn hash<H: hash::Hasher>(&self, state: &mut H) {
                self.as_bytes().hash(state)
            }
        }

        impl Ord for $t {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.as_bytes().cmp(other.as_bytes())
            }
        }

        impl PartialOrd for $t {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Eq for $t {}

        impl PartialEq for $t {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn eq(&self, other: &$t) -> bool {
                self.as_bytes().eq(other.as_bytes())
            }
        }

        impl<T> PartialEq<T> for $t
        where
            T: ?Sized + AlignedBinary,
        {
            #[inline]
            default fn eq(&self, other: &T) -> bool {
                self.as_bytes().eq(other.as_bytes())
            }
        }

        impl<T> PartialEq<Boxed<T>> for $t
        where
            T: ?Sized + AlignedBinary,
        {
            #[inline]
            fn eq(&self, other: &Boxed<T>) -> bool {
                self.as_bytes().eq(other.as_ref().as_bytes())
            }
        }

        impl<T> PartialOrd<T> for $t
        where
            T: ?Sized + AlignedBinary,
        {
            #[inline]
            default fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
                self.as_bytes().partial_cmp(other.as_bytes())
            }
        }

        impl<T> PartialOrd<Boxed<T>> for $t
        where
            T: ?Sized + AlignedBinary,
        {
            #[inline]
            fn partial_cmp(&self, other: &Boxed<T>) -> Option<core::cmp::Ordering> {
                self.as_bytes().partial_cmp(other.as_ref().as_bytes())
            }
        }
    }
}

impl_aligned_binary!(HeapBin);
impl_aligned_binary!(ProcBin);
impl_aligned_binary!(BinaryLiteral);

// We can't make this part of `impl_aligned_binary` because
// we can't implement TryInto for dynamically-sized types,
// so we implement them seperately.
macro_rules! impl_aligned_try_into {
    ($t:ty) => {
        impl TryInto<String> for $t {
            type Error = runtime::Exception;

            fn try_into(self) -> Result<String, Self::Error> {
                match str::from_utf8(self.as_bytes()) {
                    Ok(s) => Ok(s.to_owned()),
                    Err(_) => Err(badarg!()),
                }
            }
        }

        impl TryInto<String> for &$t {
            type Error = runtime::Exception;

            fn try_into(self) -> Result<String, Self::Error> {
                match str::from_utf8(self.as_bytes()) {
                    Ok(s) => Ok(s.to_owned()),
                    Err(_) => Err(badarg!()),
                }
            }
        }

        impl TryInto<Vec<u8>> for $t {
            type Error = runtime::Exception;

            #[inline]
            fn try_into(self) -> Result<Vec<u8>, Self::Error> {
                Ok(self.as_bytes().to_vec())
            }
        }

        impl TryInto<Vec<u8>> for &$t {
            type Error = runtime::Exception;

            #[inline]
            fn try_into(self) -> Result<Vec<u8>, Self::Error> {
                Ok(self.as_bytes().to_vec())
            }
        }
    }
}

impl_aligned_try_into!(ProcBin);
impl_aligned_try_into!(BinaryLiteral);

/// Displays a binary using Erlang-style formatting
pub(in super) fn display(bytes: &[u8], f: &mut fmt::Formatter) -> fmt::Result {
    match str::from_utf8(bytes) {
        Ok(s) => write!(f, "{}", s),
        Err(_) => {
            f.write_str("<<")?;

            let mut iter = bytes.iter();

            if let Some(byte) = iter.next() {
                write!(f, "{:#04x}", byte)?;

                for byte in iter {
                    write!(f, ", {:#04x}", byte)?;
                }
            }

            f.write_str(">>")
        }
    }
}

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_eq_aligned_binary_aligned_binary {
    ($o:tt for $s:tt) => {
        impl PartialEq<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn eq(&self, other: &$o) -> bool {
                self.as_bytes().eq(other.as_bytes())
            }
        }
    };
}

// No (ProcBin for HeapBin) as we always reverse order to save space
partial_eq_aligned_binary_aligned_binary!(HeapBin for ProcBin);
partial_eq_aligned_binary_aligned_binary!(HeapBin for BinaryLiteral);
partial_eq_aligned_binary_aligned_binary!(ProcBin for BinaryLiteral);

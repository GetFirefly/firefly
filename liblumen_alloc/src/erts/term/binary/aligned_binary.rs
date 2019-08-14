use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};
use core::str;

use crate::erts::term::binary::heap::HeapBin;
use crate::erts::term::binary::process::ProcBin;

/// A `BitString` that is guaranteed to always be a binary of aligned bytes
pub trait AlignedBinary {
    fn as_bytes(&self) -> &[u8];
}

pub fn display(bytes: &[u8], f: &mut fmt::Formatter) -> fmt::Result {
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

macro_rules! display_aligned_binary {
    ($t:ty) => {
        impl Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                display(self.as_bytes(), f)
            }
        }
    };
}

display_aligned_binary!(HeapBin);
display_aligned_binary!(ProcBin);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! hash_aligned_binary {
    ($t:ty) => {
        impl Hash for $t {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.as_bytes().hash(state)
            }
        }
    };
}

hash_aligned_binary!(HeapBin);
hash_aligned_binary!(ProcBin);

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

partial_eq_aligned_binary_aligned_binary!(HeapBin for HeapBin);
// No (ProcBin for HeapBin) as we always reverse order to save space
partial_eq_aligned_binary_aligned_binary!(HeapBin for ProcBin);
partial_eq_aligned_binary_aligned_binary!(ProcBin for ProcBin);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! ord_aligned_binary {
    ( $s:tt) => {
        impl Ord for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.as_bytes().cmp(other.as_bytes())
            }
        }

        impl PartialOrd for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
    };
}

ord_aligned_binary!(HeapBin);
ord_aligned_binary!(ProcBin);

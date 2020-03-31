use core::cmp;

use super::prelude::*;
use crate::erts::term::prelude::Boxed;

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_eq_aligned_binary_maybe_aligned_maybe_binary {
    ($o:tt for $s:tt) => {
        impl PartialEq<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn eq(&self, other: &$o) -> bool {
                if self.is_binary() {
                    if self.is_aligned() {
                        unsafe { self.as_bytes_unchecked() }.eq(other.as_bytes())
                    } else {
                        self.full_byte_iter().eq(other.full_byte_iter())
                    }
                } else {
                    false
                }
            }
        }

        impl PartialEq<Boxed<$o>> for $s {
            #[inline]
            fn eq(&self, other: &Boxed<$o>) -> bool {
                self.eq(other.as_ref())
            }
        }
        impl PartialEq<Boxed<$s>> for $o {
            #[inline]
            fn eq(&self, other: &Boxed<$s>) -> bool {
                other.as_ref().eq(self)
            }
        }
    };
}

partial_eq_aligned_binary_maybe_aligned_maybe_binary!(HeapBin for MatchContext);
partial_eq_aligned_binary_maybe_aligned_maybe_binary!(ProcBin for MatchContext);
partial_eq_aligned_binary_maybe_aligned_maybe_binary!(BinaryLiteral for MatchContext);
partial_eq_aligned_binary_maybe_aligned_maybe_binary!(HeapBin for SubBinary);
partial_eq_aligned_binary_maybe_aligned_maybe_binary!(ProcBin for SubBinary);
partial_eq_aligned_binary_maybe_aligned_maybe_binary!(BinaryLiteral for SubBinary);

// Has to have explicit types to prevent E0119: conflicting implementations of trait
macro_rules! partial_ord_aligned_binary_maybe_aligned_maybe_binary {
    ($o:tt for $s:tt) => {
        impl PartialOrd<$o> for $s {
            /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
            /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
            fn partial_cmp(&self, other: &$o) -> Option<cmp::Ordering> {
                use cmp::Ordering::*;

                let mut self_full_byte_iter = self.full_byte_iter();
                let mut other_full_byte_iter = other.full_byte_iter();
                let mut partial_ordering = Some(Equal);

                while let Some(Equal) = partial_ordering {
                    match (self_full_byte_iter.next(), other_full_byte_iter.next()) {
                        (Some(self_byte), Some(other_byte)) => {
                            partial_ordering = self_byte.partial_cmp(&other_byte)
                        }
                        (None, Some(other_byte)) => {
                            let partial_byte_bit_len = self.partial_byte_bit_len();

                            partial_ordering =
                                if partial_byte_bit_len > 0 {
                                    self.partial_byte_bit_iter().partial_cmp(
                                        BitsIter::new_with_max(other_byte, partial_byte_bit_len),
                                    )
                                } else {
                                    Some(Less)
                                };

                            break;
                        }
                        (Some(_), None) => {
                            partial_ordering = Some(Greater);

                            break;
                        }
                        (None, None) => {
                            if 0 < self.partial_byte_bit_len() {
                                partial_ordering = Some(Greater);
                            }

                            break;
                        }
                    }
                }

                partial_ordering
            }
        }

        impl PartialOrd<Boxed<$o>> for $s {
            #[inline]
            fn partial_cmp(&self, other: &Boxed<$o>) -> Option<cmp::Ordering> {
                self.partial_cmp(other.as_ref())
            }
        }

        impl PartialOrd<Boxed<$s>> for $o {
            #[inline]
            fn partial_cmp(&self, other: &Boxed<$s>) -> Option<cmp::Ordering> {
                other.as_ref().partial_cmp(self).map(|o| o.reverse())
            }
        }
    };
}

partial_ord_aligned_binary_maybe_aligned_maybe_binary!(HeapBin for MatchContext);
partial_ord_aligned_binary_maybe_aligned_maybe_binary!(ProcBin for MatchContext);
partial_ord_aligned_binary_maybe_aligned_maybe_binary!(BinaryLiteral for MatchContext);
partial_ord_aligned_binary_maybe_aligned_maybe_binary!(HeapBin for SubBinary);
partial_ord_aligned_binary_maybe_aligned_maybe_binary!(ProcBin for SubBinary);
partial_ord_aligned_binary_maybe_aligned_maybe_binary!(BinaryLiteral for SubBinary);

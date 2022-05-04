///! This module defines a trait which represents the idea of equality without
///! the possibility of coercion. Its usage in Lumen is to extend the behavior
///! of `Eq` for terms to respect the semantics of the `=:=` and `=/=` operators.
use core::cmp::PartialEq;

/// The default implementation of this trait simply delegates to `Eq`, override
/// the `exact_eq` or `exact_ne` methods to extend that behavior.
pub trait ExactEq: PartialEq<Self> {
    fn exact_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }

    fn exact_ne(&self, other: &Self) -> bool {
        !self.exact_eq(other)
    }
}

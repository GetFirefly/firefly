///! This module defines a trait which represents the idea of equality without
///! the possibility of coercion. Its usage in Firefly is to extend the behavior
///! of `Eq` for terms to respect the semantics of the `=:=` and `=/=` operators.

/// This trait extends the notion of equality provided by `Eq` to cover the distinction
/// between the non-strict (`==` and `/=`), and strict (`=:=` and `=/=`) equality operators.
///
/// ExactEq has a default implemention for all `Eq` implementors. However, for types which distinguish
/// between strict/non-strict equality, this trait should be specialized on those types.
pub trait ExactEq: Eq {
    fn exact_eq(&self, other: &Self) -> bool;

    fn exact_ne(&self, other: &Self) -> bool;
}

impl<T: ?Sized + Eq> ExactEq for T {
    default fn exact_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }

    default fn exact_ne(&self, other: &Self) -> bool {
        self.ne(other)
    }
}

///! This module defines a trait which represents the idea of equality without
///! the possibility of coercion. Its usage in Firefly is to extend the behavior
///! of `Eq` for terms to respect the semantics of the `=:=` and `=/=` operators.
///!
///! This trait extends the notion of equality provided by `Eq` to cover the distinction
///! between the non-strict (`==` and `/=`), and strict (`=:=` and `=/=`) equality operators.
///!
///! ExactEq has a default implemention for all `Eq` implementors. However, for types which distinguish
///! between strict/non-strict equality, this trait should be specialized on those types.
use alloc::alloc::Allocator;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use firefly_number::{Float, Int, Number};

use crate::gc::Gc;

/// This trait implies precise equality between two terms, i.e. no coercion
/// between types. By default, an implementation is provided for all types
/// that implement `Eq`, using the associated `PartialEq` implementation, but
/// this trait should be implemented for all term types, or container types that
/// carry term types, to ensure the semantics of comparisons in Erlang are preserved.
pub trait ExactEq: Eq {
    fn exact_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }

    fn exact_ne(&self, other: &Self) -> bool {
        !self.exact_eq(other)
    }
}

impl crate::cmp::ExactEq for Number {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Number::Integer(x), Number::Integer(y)) => x.eq(y),
            (Number::Float(x), Number::Float(y)) => x.eq(y),
            _ => false,
        }
    }
}

impl crate::cmp::ExactEq for Float {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl crate::cmp::ExactEq for Int {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl<T: ExactEq, A: Allocator> ExactEq for Box<T, A> {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        ExactEq::exact_eq(&**self, &**other)
    }
}

impl<T: ExactEq + Eq + PartialEq<Self>> ExactEq for Gc<T> {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        ExactEq::exact_eq(&**self, &**other)
    }
}

impl<T: ExactEq> ExactEq for Arc<T> {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        ExactEq::exact_eq(&**self, &**other)
    }
}

impl<T: ExactEq> ExactEq for [T] {
    fn exact_eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(x, y)| x.exact_eq(&y))
    }
}

impl<T: ExactEq, A: Allocator> ExactEq for Vec<T, A> {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        self.as_slice().exact_eq(other.as_slice())
    }
}

impl<T: ExactEq, const N: usize> ExactEq for [T; N] {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        self[..].exact_eq(&other[..])
    }
}

mod hooks;
mod stats_alloc;

pub use self::stats_alloc::StatsAlloc;

use core::cmp::Ordering;
use core::hash;

use alloc::vec::Vec;

use num_traits::ToPrimitive;

pub use histogram::Histogram;
pub use minmax::MinMax;
pub use online::{mean, stddev, variance, OnlineStats};

/// Partial wraps a type that satisfies `PartialOrd` and implements `Ord`.
///
/// This allows types like `f64` to be used in data structures that require
/// `Ord`. When an ordering is not defined, an arbitrary order is returned.
#[derive(Clone, PartialEq, PartialOrd)]
struct Partial<T>(pub T);

impl<T: PartialEq> Eq for Partial<T> {}

impl<T: PartialOrd> Ord for Partial<T> {
    fn cmp(&self, other: &Partial<T>) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}
impl<T: ToPrimitive> ToPrimitive for Partial<T> {
    fn to_isize(&self) -> Option<isize> {
        self.0.to_isize()
    }
    fn to_i8(&self) -> Option<i8> {
        self.0.to_i8()
    }
    fn to_i16(&self) -> Option<i16> {
        self.0.to_i16()
    }
    fn to_i32(&self) -> Option<i32> {
        self.0.to_i32()
    }
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_usize(&self) -> Option<usize> {
        self.0.to_usize()
    }
    fn to_u8(&self) -> Option<u8> {
        self.0.to_u8()
    }
    fn to_u16(&self) -> Option<u16> {
        self.0.to_u16()
    }
    fn to_u32(&self) -> Option<u32> {
        self.0.to_u32()
    }
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }
    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

impl<T: hash::Hash> hash::Hash for Partial<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Defines an interface for types that have an identity and can be commuted.
///
/// The value returned by `Default::default` must be its identity with respect
/// to the `merge` operation.
pub trait Commute: Sized {
    /// Merges the value `other` into `self`.
    fn merge(&mut self, other: Self);

    /// Merges the values in the iterator into `self`.
    fn consume<I: Iterator<Item = Self>>(&mut self, other: I) {
        for v in other {
            self.merge(v);
        }
    }
}

/// Merges all items in the stream.
///
/// If the stream is empty, `None` is returned.
pub fn merge_all<T: Commute, I: Iterator<Item = T>>(mut it: I) -> Option<T> {
    match it.next() {
        None => None,
        Some(mut init) => {
            init.consume(it);
            Some(init)
        }
    }
}

impl<T: Commute> Commute for Option<T> {
    fn merge(&mut self, other: Option<T>) {
        match *self {
            None => {
                *self = other;
            }
            Some(ref mut v1) => {
                other.map(|v2| v1.merge(v2));
            }
        }
    }
}

impl<T: Commute, E> Commute for Result<T, E> {
    fn merge(&mut self, other: Result<T, E>) {
        // Can't figure out how to work around the borrow checker to make
        // this code less awkward.
        if !self.is_err() && other.is_err() {
            *self = other;
            return;
        }
        match *self {
            Err(_) => {}
            Ok(ref mut v1) => {
                match other {
                    Ok(v2) => {
                        v1.merge(v2);
                    }
                    // This is the awkward part. We can't assign to `*self`
                    // because of the `ref mut v1` borrow. So we catch this
                    // case above and declare that this cannot be reached.
                    Err(_) => {
                        unreachable!();
                    }
                }
            }
        }
    }
}

impl<T: Commute> Commute for Vec<T> {
    fn merge(&mut self, other: Vec<T>) {
        assert_eq!(self.len(), other.len());
        for (v1, v2) in self.iter_mut().zip(other.into_iter()) {
            v1.merge(v2);
        }
    }
}

mod histogram;
mod minmax;
mod online;

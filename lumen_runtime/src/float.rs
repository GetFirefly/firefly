use std::hash::{Hash, Hasher};
#[cfg(test)]
use std::ops::RangeInclusive;

use crate::heap::{CloneIntoHeap, Heap};
use crate::term::{Tag, Term};

pub const INTEGRAL_MIN: f64 = -9007199254740992.0;
pub const INTEGRAL_MAX: f64 = 9007199254740992.0;

pub struct Float {
    #[allow(dead_code)]
    header: Term,
    pub inner: f64,
}

impl Float {
    pub fn new(overflowing_inner: f64) -> Self {
        Float {
            header: Term {
                tagged: Tag::Float as usize,
            },
            inner: Self::clamp_inner(overflowing_inner),
        }
    }

    #[cfg(test)]
    pub fn clamp_inclusive_range(overflowing_range: RangeInclusive<f64>) -> RangeInclusive<f64> {
        Self::clamp_inner(overflowing_range.start().clone())
            ..=Self::clamp_inner(overflowing_range.end().clone())
    }

    fn clamp_inner(overflowing: f64) -> f64 {
        if overflowing == std::f64::NEG_INFINITY {
            std::f64::MIN
        } else if overflowing == std::f64::INFINITY {
            std::f64::MAX
        } else {
            overflowing
        }
    }
}

impl CloneIntoHeap for &'static Float {
    fn clone_into_heap(&self, heap: &Heap) -> &'static Float {
        heap.f64_to_float(self.inner)
    }
}

impl Eq for Float {}

impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.to_bits().hash(state)
    }
}

impl PartialEq for Float {
    fn eq(&self, other: &Float) -> bool {
        self.inner == other.inner
    }
}

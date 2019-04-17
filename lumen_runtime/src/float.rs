use std::hash::{Hash, Hasher};

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

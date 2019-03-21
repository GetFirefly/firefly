use std::hash::{Hash, Hasher};

use crate::term::{Tag, Term};

pub struct Float {
    #[allow(dead_code)]
    header: Term,
    pub inner: f64,
}

impl Float {
    pub fn new(inner: f64) -> Self {
        Float {
            header: Term {
                tagged: Tag::Float as usize,
            },
            inner,
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

    fn ne(&self, other: &Float) -> bool {
        !self.eq(other)
    }
}

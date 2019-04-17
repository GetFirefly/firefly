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

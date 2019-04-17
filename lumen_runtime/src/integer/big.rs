use crate::term::{Tag, Term};

pub struct Integer {
    #[allow(dead_code)]
    header: Term,
    pub inner: rug::Integer,
}

impl Integer {
    pub fn new(inner: rug::Integer) -> Self {
        Integer {
            header: Term {
                tagged: Tag::BigInteger as usize,
            },
            inner,
        }
    }
}

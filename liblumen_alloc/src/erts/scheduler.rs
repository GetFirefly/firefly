pub mod id;

#[derive(Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ID(pub id::Raw);

impl ID {
    pub fn new(raw: id::Raw) -> ID {
        ID(raw)
    }
}

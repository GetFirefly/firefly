#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Node {
    id: usize,
}

impl Node {
    pub(in crate::erts) fn new(id: usize) -> Self {
        Self { id }
    }
}

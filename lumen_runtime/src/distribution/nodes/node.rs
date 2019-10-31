use std::sync::Arc;

use liblumen_alloc::erts::term::{AsTerm, Atom, Term};
use liblumen_alloc::erts::Node;

pub const DEAD_ATOM_NAME: &str = "nonode@nohost";

pub fn dead_atom() -> Atom {
    Atom::try_from_str(DEAD_ATOM_NAME).unwrap()
}

pub fn arc_node() -> Arc<Node> {
    ARC_NODE.clone()
}

pub fn atom() -> Atom {
    ARC_NODE.name()
}

pub fn id() -> usize {
    ARC_NODE.id()
}

pub fn term() -> Term {
    unsafe { atom().as_term() }
}

lazy_static! {
    pub(super) static ref ARC_NODE: Arc<Node> = Arc::new(Node::new(ID, dead_atom(), CREATION));
}

const CREATION: u32 = 0;
const ID: usize = 0;

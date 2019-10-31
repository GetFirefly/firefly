use core::cell::Cell;
use core::cmp::{self, Ord, PartialEq, PartialOrd};
use core::hash::{Hash, Hasher};

use liblumen_core::locks::Mutex;

use crate::erts::term::Atom;

#[derive(Debug)]
pub struct Node {
    id: usize,
    name: Mutex<Cell<Atom>>,
    creation: u32,
}

impl Node {
    pub fn new(id: usize, name: Atom, creation: u32) -> Self {
        Self {
            id,
            name: Mutex::new(Cell::new(name)),
            creation,
        }
    }

    pub fn creation(&self) -> u32 {
        self.creation
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn name(&self) -> Atom {
        self.name.lock().get()
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // name is not included as a node should remain consistent if it changes its name, such as
        // the running node going from dead to alive.
        self.id.hash(state);
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Node) -> cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialEq<Node> for Node {
    fn eq(&self, other: &Node) -> bool {
        self.id == other.id
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Node) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

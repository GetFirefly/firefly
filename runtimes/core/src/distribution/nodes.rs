pub mod node;

use std::backtrace::Backtrace;
use std::sync::Arc;

use hashbrown::HashMap;
use lazy_static::lazy_static;
use thiserror::Error;

use liblumen_core::locks::RwLock;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::{ArcError, Exception, InternalException, RuntimeException};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Node;

pub fn atom_to_arc_node(atom: &Atom) -> Option<Arc<Node>> {
    RW_LOCK_ARC_NODE_BY_NAME
        .read()
        .get(atom)
        .map(|ref_arc_node| ref_arc_node.clone())
}

pub fn try_atom_to_arc_node(atom: &Atom) -> Result<Arc<Node>, NodeNotFound> {
    match atom_to_arc_node(atom) {
        Some(arc_node) => Ok(arc_node),
        None => Err(NodeNotFound::Name {
            name: atom.clone(),
            backtrace: Backtrace::capture(),
        }),
    }
}

pub fn id_to_arc_node(id: &usize) -> Option<Arc<Node>> {
    RW_LOCK_ARC_NODE_BY_ID
        .read()
        .get(id)
        .map(|ref_arc_node| ref_arc_node.clone())
}

pub fn try_id_to_arc_node(id: &usize) -> Result<Arc<Node>, NodeNotFound> {
    match id_to_arc_node(id) {
        Some(arc_node) => Ok(arc_node),
        None => Err(NodeNotFound::ID {
            id: *id,
            backtrace: Backtrace::capture(),
        }),
    }
}

pub fn insert(arc_node: Arc<Node>) {
    let id = arc_node.id();
    let name = arc_node.name();

    let mut arc_node_by_id = RW_LOCK_ARC_NODE_BY_ID.write();
    let mut arc_node_by_name = RW_LOCK_ARC_NODE_BY_NAME.write();

    if let Some(id_arc_node) = arc_node_by_id.remove(&id) {
        if let Some(id_name_arc_node) = arc_node_by_name.remove(&id_arc_node.name()) {
            assert_eq!(id_name_arc_node.id(), id);
        }
    }

    if let Some(name_arc_node) = arc_node_by_name.remove(&name) {
        if let Some(name_id_arc_node) = arc_node_by_id.remove(&name_arc_node.id()) {
            assert_eq!(name_id_arc_node.name(), name);
        }
    }

    arc_node_by_id
        .insert(arc_node.id(), arc_node.clone())
        .unwrap_none();
    arc_node_by_name
        .insert(arc_node.name(), arc_node)
        .unwrap_none();
}

#[derive(Debug, Error)]
pub enum NodeNotFound {
    #[error("No node with name ({name})")]
    Name { name: Atom, backtrace: Backtrace },
    #[error("No node with id ({id})")]
    ID { id: usize, backtrace: Backtrace },
}

impl From<NodeNotFound> for Exception {
    fn from(node_not_found: NodeNotFound) -> Self {
        RuntimeException::from(node_not_found).into()
    }
}
impl From<NodeNotFound> for InternalException {
    fn from(node_not_found: NodeNotFound) -> Self {
        ArcError::from_err(node_not_found).into()
    }
}
impl From<NodeNotFound> for RuntimeException {
    fn from(node_not_found: NodeNotFound) -> Self {
        badarg!(ArcError::from_err(node_not_found))
    }
}

lazy_static! {
    static ref RW_LOCK_ARC_NODE_BY_ID: RwLock<HashMap<usize, Arc<Node>>> = {
        let mut hash_map = HashMap::new();
        let arc_node = node::arc_node();
        hash_map.insert(arc_node.id(), arc_node);

        RwLock::new(hash_map)
    };
    static ref RW_LOCK_ARC_NODE_BY_NAME: RwLock<HashMap<Atom, Arc<Node>>> = {
        let mut hash_map = HashMap::new();
        let arc_node = node::arc_node();
        hash_map.insert(arc_node.name(), arc_node);

        RwLock::new(hash_map)
    };
}

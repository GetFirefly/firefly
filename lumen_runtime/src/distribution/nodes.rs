pub mod node;

use std::sync::Arc;

use hashbrown::HashMap;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Node;

pub fn atom_to_arc_node(atom: &Atom) -> Option<Arc<Node>> {
    RW_LOCK_ARC_NODE_BY_NAME
        .read()
        .get(atom)
        .map(|ref_arc_node| ref_arc_node.clone())
}

pub fn id_to_arc_node(id: &usize) -> Option<Arc<Node>> {
    RW_LOCK_ARC_NODE_BY_ID
        .read()
        .get(id)
        .map(|ref_arc_node| ref_arc_node.clone())
}

// TODO make non-test-only once distribution connection is implemented
#[cfg(all(not(target_arch = "wasm32"), test))]
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

use std::sync::Arc;

use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::{badarg, Node};

use crate::distribution::nodes::atom_to_arc_node;

use super::atom;

pub fn decode(safe: bool, bytes: &[u8]) -> Result<(Arc<Node>, &[u8]), Exception> {
    let (atom, after_atom_bytes) = atom::decode_tagged(safe, bytes)?;

    match atom_to_arc_node(&atom) {
        Some(arc_node) => Ok((arc_node, after_atom_bytes)),
        None => Err(badarg!().into()),
    }
}

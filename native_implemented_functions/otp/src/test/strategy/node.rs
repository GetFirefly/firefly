mod atom;
mod id;

use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::Node;

use crate::runtime::distribution::nodes;

pub fn external() -> BoxedStrategy<Arc<Node>> {
    (id::external(), atom::external(), any::<u32>())
        .prop_map(|(id, atom, creation)| {
            let arc_node = Arc::new(Node::new(id, atom, creation));

            nodes::insert(arc_node.clone());

            arc_node
        })
        .boxed()
}

fn id() -> BoxedStrategy<usize> {
    any::<usize>().boxed()
}

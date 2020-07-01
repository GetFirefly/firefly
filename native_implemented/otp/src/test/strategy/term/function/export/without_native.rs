use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    (super::module_atom(), super::function())
        .prop_map(move |(module, function)| {
            arc_process
                .export_closure(module, function, arity, None)
                .unwrap()
        })
        .boxed()
}

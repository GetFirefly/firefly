use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    (
        super::super::module_atom(),
        super::index(),
        super::old_unique(),
        super::unique(),
        super::creator(),
    )
        .prop_map(move |(module, index, old_unique, unique, creator)| {
            arc_process
                .anonymous_closure_with_env_from_slice(
                    module,
                    index,
                    old_unique,
                    unique,
                    arity,
                    None,
                    creator,
                    &[],
                )
                .unwrap()
        })
        .boxed()
}

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use crate::test::strategy;

pub fn function() -> BoxedStrategy<Atom> {
    strategy::atom()
}

pub fn with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    (
        super::module_atom(),
        function(),
        super::option_located_code(),
    )
        .prop_map(move |(module, function, option_located_code)| {
            let definition = Definition::Export { function };

            if let Some(located_code) = option_located_code {
                crate::code::insert(module, definition.clone(), arity, located_code);
            }

            arc_process
                .closure_with_env_from_slice(module, definition, arity, option_located_code, &[])
                .unwrap()
        })
        .boxed()
}

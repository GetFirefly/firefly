use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::prelude::{Atom, Term};
use liblumen_alloc::erts::Process;

use crate::test::strategy;

pub fn function() -> BoxedStrategy<Atom> {
    strategy::atom()
}

pub fn with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    (
        super::module_atom(),
        function(),
        super::option_debuggable_code(),
    )
        .prop_map(move |(module, function, option_debuggable_code)| {
            let option_code = option_debuggable_code.map(|debuggable_code| debuggable_code.0);

            if let Some(code) = option_code {
                crate::runtime::code::export::insert(module, function, arity, code);
            }

            arc_process
                .export_closure(module, function, arity, option_code)
                .unwrap()
        })
        .boxed()
}

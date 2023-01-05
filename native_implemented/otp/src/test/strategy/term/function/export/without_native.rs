use std::sync::Arc;

use proptest::strategy::BoxedStrategy;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

pub fn with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    (super::module_atom(), super::function())
        .prop_map(move |(module, function)| {
            arc_process.export_closure(module, function, arity, None)
        })
        .boxed()
}

use std::sync::Arc;

use proptest::strategy::BoxedStrategy;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

pub fn with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    (
        super::super::module_atom(),
        super::index(),
        super::old_unique(),
        super::unique(),
    )
        .prop_map(move |(module, index, old_unique, unique)| {
            arc_process.anonymous_closure_with_env_from_slice(
                module,
                index,
                old_unique,
                unique,
                arity,
                None,
                &[],
            )
        })
        .boxed()
}

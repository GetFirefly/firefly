use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Strategy};

use firefly_rt::process::Process;
use firefly_rt::term::Term;

pub fn intermediate(
    key_or_value: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::hash_map(key_or_value.clone(), key_or_value, size_range)
        .prop_map(move |mut hash_map| {
            let entry_vec: Vec<(Term, Term)> = hash_map.drain().collect();

            arc_process.map_from_slice(&entry_vec)
        })
        .boxed()
}

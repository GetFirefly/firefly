use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::map_size_1::result;
use crate::test::strategy;

#[test]
fn without_map_errors_badmap() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_map(arc_process.clone()),
            )
        },
        |(arc_process, map)| {
            prop_assert_badmap!(
                result(&arc_process, map),
                &arc_process,
                map,
                format!("map ({}) is not a map", map)
            );

            Ok(())
        },
    );
}

#[test]
fn with_map_returns_number_of_entries() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                proptest::collection::hash_map(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process),
                    strategy::size_range(),
                ),
            )
                .prop_map(|(arc_process, mut hash_map)| {
                    let entry_vec: Vec<(Term, Term)> = hash_map.drain().collect();

                    (
                        arc_process.clone(),
                        arc_process.map_from_slice(&entry_vec).unwrap(),
                        arc_process.integer(entry_vec.len()).unwrap(),
                    )
                })
        },
        |(arc_process, map, size)| {
            prop_assert_eq!(result(&arc_process, map,), Ok(size));

            Ok(())
        },
    );
}

use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badmap;
use liblumen_alloc::erts::term::prelude::Term;(

use crate::otp::erlang::map_size_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_map_errors_badmap() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_map(arc_process.clone()), |map| {
                prop_assert_eq!(native(&arc_process, map,), Err(badmap!(&arc_process, map)));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_map_returns_number_of_entries() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| {
                    (
                        Just(arc_process.clone()),
                        proptest::collection::hash_map(
                            strategy::term(arc_process.clone()),
                            strategy::term(arc_process),
                            strategy::size_range(),
                        ),
                    )
                })
                .prop_map(|(arc_process, mut hash_map)| {
                    let entry_vec: Vec<(Term, Term)> = hash_map.drain().collect();

                    (
                        arc_process.clone(),
                        arc_process.map_from_slice(&entry_vec).unwrap(),
                        arc_process.integer(entry_vec.len()).unwrap(),
                    )
                }),
            |(arc_process, map, size)| {
                prop_assert_eq!(native(&arc_process, map,), Ok(size));

                Ok(())
            },
        )
        .unwrap();
}

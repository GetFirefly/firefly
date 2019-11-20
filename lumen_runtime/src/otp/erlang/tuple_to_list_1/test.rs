use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::erlang::tuple_to_list_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_tuple_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::is_not_tuple(arc_process),
                )
            }),
            |(arc_process, tuple)| {
                prop_assert_eq!(
                    native(&arc_process, tuple),
                    Err(badarg!(&arc_process).into())
                );

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_returns_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &proptest::collection::vec(strategy::term(arc_process.clone()), 0..=3),
                |element_vec| {
                    let tuple = arc_process.tuple_from_slice(&element_vec).unwrap();
                    let list = arc_process.list_from_slice(&element_vec).unwrap();

                    prop_assert_eq!(native(&arc_process, tuple), Ok(list));

                    Ok(())
                },
            )
            .unwrap();
    });
}

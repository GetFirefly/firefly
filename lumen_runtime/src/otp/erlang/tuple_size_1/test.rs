use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::erlang::tuple_size_1::native;
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
fn with_tuple_returns_arity() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(0_usize..=3_usize).prop_flat_map(|size| {
                    (
                        Just(size),
                        strategy::term::tuple::intermediate(
                            strategy::term(arc_process.clone()),
                            (size..=size).into(),
                            arc_process.clone(),
                        ),
                    )
                }),
                |(size, term)| {
                    prop_assert_eq!(
                        native(&arc_process, term),
                        Ok(arc_process.integer(size).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

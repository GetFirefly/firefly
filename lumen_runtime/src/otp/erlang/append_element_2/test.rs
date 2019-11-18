use std::convert::TryInto;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::{Boxed, Tuple};

use crate::otp::erlang::append_element_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_tuple_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term(arc_process),
                )
            }),
            |(arc_process, tuple, element)| {
                prop_assert_eq!(native(&arc_process, tuple, element), Err(badarg!().into()));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_returns_tuple_with_new_element_at_end() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(tuple, element)| {
                    let result = native(&arc_process, tuple, element);

                    prop_assert!(result.is_ok(), "{:?}", result);

                    let appended_tuple = result.unwrap();

                    let appended_tuple_tuple_result: core::result::Result<Boxed<Tuple>, _> =
                        appended_tuple.try_into();

                    prop_assert!(appended_tuple_tuple_result.is_ok());

                    let appended_tuple_tuple = appended_tuple_tuple_result.unwrap();
                    let tuple_tuple: Boxed<Tuple> = tuple.try_into().unwrap();

                    prop_assert_eq!(appended_tuple_tuple.len(), tuple_tuple.len() + 1);
                    prop_assert_eq!(appended_tuple_tuple[tuple_tuple.len()], element);

                    Ok(())
                },
            )
            .unwrap();
    });
}

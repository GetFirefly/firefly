use std::convert::TryInto;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::{Boxed, Tuple};

use crate::otp::erlang::make_tuple_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_arity_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_arity(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(arity, initial_value)| {
                    prop_assert_eq!(
                        native(&arc_process, arity, initial_value),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_arity_returns_tuple_with_arity_copies_of_initial_value() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    (0_usize..255_usize),
                    strategy::term(arc_process),
                )
            }),
            |(arc_process, arity_usize, initial_value)| {
                let arity = arc_process.integer(arity_usize).unwrap();

                let result = native(&arc_process, arity, initial_value);

                prop_assert!(result.is_ok());

                let tuple_term = result.unwrap();

                prop_assert!(tuple_term.is_tuple());

                let boxed_tuple: Boxed<Tuple> = tuple_term.try_into().unwrap();

                prop_assert_eq!(boxed_tuple.len(), arity_usize);

                for element in boxed_tuple.iter() {
                    prop_assert_eq!(element, initial_value);
                }

                Ok(())
            },
        )
        .unwrap();
}

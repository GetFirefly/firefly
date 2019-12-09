use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::setelement_3::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_tuple_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term(arc_process),
                )
            }),
            |(arc_process, tuple, index, element)| {
                prop_assert_badarg!(
                    native(&arc_process, index, tuple, element),
                    format!("tuple ({}) must be a tuple", tuple)
                );

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_without_valid_index_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple::without_index(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |((tuple, index), element)| {
                    let boxed_tuple: Boxed<Tuple> = tuple.try_into().unwrap();

                    prop_assert_badarg!(
                        native(&arc_process, index, tuple, element),
                        format!(
                            "index ({}) must be a 1-based index in 1-{}",
                            index,
                            boxed_tuple.len()
                        )
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_with_valid_index_returns_tuple_with_index_replaced() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple::with_index(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |((mut element_vec, element_vec_index, tuple, index), element)| {
                    element_vec[element_vec_index] = element;
                    let new_tuple = arc_process.tuple_from_slice(&element_vec).unwrap();

                    prop_assert_eq!(native(&arc_process, index, tuple, element), Ok(new_tuple));

                    Ok(())
                },
            )
            .unwrap();
    });
}

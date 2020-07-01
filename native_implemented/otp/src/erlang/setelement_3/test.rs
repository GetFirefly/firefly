use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::setelement_3::result;
use crate::test::strategy;

#[test]
fn without_tuple_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::is_not_tuple(arc_process.clone()),
                strategy::term(arc_process),
            )
        },
        |(arc_process, tuple, index, element)| {
            prop_assert_is_not_tuple!(result(&arc_process, index, tuple, element), tuple);

            Ok(())
        },
    );
}

#[test]
fn with_tuple_without_valid_index_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::tuple::without_index(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, (tuple, index), element)| {
            let boxed_tuple: Boxed<Tuple> = tuple.try_into().unwrap();

            prop_assert_badarg!(
                result(&arc_process, index, tuple, element),
                format!(
                    "index ({}) is not a 1-based integer between 1-{}",
                    index,
                    boxed_tuple.len()
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_tuple_with_valid_index_returns_tuple_with_index_replaced() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::tuple::with_index(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, (mut element_vec, element_vec_index, tuple, index), element)| {
            element_vec[element_vec_index] = element;
            let new_tuple = arc_process.tuple_from_slice(&element_vec).unwrap();

            prop_assert_eq!(result(&arc_process, index, tuple, element), Ok(new_tuple));

            Ok(())
        },
    );
}

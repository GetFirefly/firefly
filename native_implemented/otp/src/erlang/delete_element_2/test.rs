use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::delete_element_2::result;
use crate::test::strategy;

#[test]
fn without_tuple_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_tuple(arc_process.clone()),
                strategy::term::is_integer(arc_process),
            )
        },
        |(arc_process, tuple, index)| {
            prop_assert_is_not_tuple!(result(&arc_process, index, tuple), tuple);

            Ok(())
        },
    );
}

#[test]
fn with_tuple_without_integer_between_1_and_the_length_inclusive_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::tuple::without_index(arc_process.clone()),
            )
        },
        |(arc_process, (tuple, index))| {
            let tuple_tuple: Boxed<Tuple> = tuple.try_into().unwrap();

            prop_assert_badarg!(
                result(&arc_process, index, tuple),
                format!(
                    "index ({}) is not a 1-based integer between 1-{}",
                    index,
                    tuple_tuple.len()
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_tuple_with_integer_between_1_and_the_length_inclusive_returns_tuple_without_element() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::tuple::with_index(arc_process.clone()),
            )
        },
        |(arc_process, (mut element_vec, element_vec_index, tuple, index))| {
            element_vec.remove(element_vec_index);

            prop_assert_eq!(
                result(&arc_process, index, tuple),
                Ok(arc_process.tuple_from_slice(&element_vec).unwrap())
            );

            Ok(())
        },
    );
}

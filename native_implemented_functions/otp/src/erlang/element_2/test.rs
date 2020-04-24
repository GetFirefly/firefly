use proptest::prop_assert_eq;

use crate::erlang::element_2::result;
use crate::test::strategy;

#[test]
fn without_tuple_errors_badarg() {
    run!(
        |arc_process| {
            (
                strategy::term::is_not_tuple(arc_process.clone()),
                strategy::term::is_integer(arc_process),
            )
        },
        |(tuple, index)| {
            prop_assert_is_not_tuple!(result(index, tuple), tuple);

            Ok(())
        },
    );
}

#[test]
fn with_tuple_without_integer_between_1_and_the_length_inclusive_errors_badarg() {
    run!(
        |arc_process| strategy::term::tuple::without_index(arc_process.clone()),
        |(tuple, index)| {
            prop_assert_badarg!(result(index, tuple), format!("index ({})", index));

            Ok(())
        },
    );
}

#[test]
fn with_tuple_with_integer_between_1_and_the_length_inclusive_returns_tuple_without_element() {
    run!(
        |arc_process| strategy::term::tuple::with_index(arc_process.clone()),
        |(element_vec, element_vec_index, tuple, index)| {
            prop_assert_eq!(result(index, tuple), Ok(element_vec[element_vec_index]));

            Ok(())
        },
    );
}

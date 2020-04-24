use std::convert::TryInto;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::{Boxed, Tuple};

use crate::erlang::append_element_2::result;
use crate::test::strategy;

#[test]
fn without_tuple_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_tuple(arc_process.clone()),
                strategy::term(arc_process),
            )
        },
        |(arc_process, tuple, element)| {
            prop_assert_is_not_tuple!(result(&arc_process, tuple, element), tuple);

            Ok(())
        },
    );
}

#[test]
fn with_tuple_returns_tuple_with_new_element_at_end() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::tuple(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, tuple, element)| {
            let result = result(&arc_process, tuple, element);

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
    );
}

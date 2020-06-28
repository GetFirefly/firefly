use std::convert::TryInto;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::make_tuple_2::result;
use crate::test::strategy;

#[test]
fn without_arity_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_arity(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, arity, initial_value)| {
            prop_assert_is_not_arity!(result(&arc_process, arity, initial_value), arity);

            Ok(())
        },
    );
}

#[test]
fn with_arity_returns_tuple_with_arity_copies_of_initial_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                (0_usize..255_usize),
                strategy::term(arc_process),
            )
        },
        |(arc_process, arity_usize, initial_value)| {
            let arity = arc_process.integer(arity_usize).unwrap();

            let result = result(&arc_process, arity, initial_value);

            prop_assert!(result.is_ok());

            let tuple_term = result.unwrap();

            prop_assert!(tuple_term.is_boxed());

            let boxed_tuple: Result<Boxed<Tuple>, _> = tuple_term.try_into();
            prop_assert!(boxed_tuple.is_ok());

            let tuple = boxed_tuple.unwrap();

            prop_assert_eq!(tuple.len(), arity_usize);

            for element in tuple.iter() {
                prop_assert_eq!(element, &initial_value);
            }

            Ok(())
        },
    );
}

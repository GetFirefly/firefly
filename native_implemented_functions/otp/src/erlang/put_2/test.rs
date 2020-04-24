use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::put_2::result;
use crate::test::strategy;

#[test]
fn without_key_returns_undefined_for_previous_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, key, value)| {
            arc_process.erase_entries().unwrap();

            prop_assert_eq!(
                result(&arc_process, key, value),
                Ok(Atom::str_to_term("undefined"))
            );

            prop_assert_eq!(arc_process.get_value_from_key(key), value);

            Ok(())
        },
    );
}

#[test]
fn with_key_returns_previous_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process),
            )
        },
        |(arc_process, key, old_value, new_value)| {
            arc_process.erase_entries().unwrap();

            arc_process.put(key, old_value).unwrap();

            prop_assert_eq!(result(&arc_process, key, new_value), Ok(old_value));

            prop_assert_eq!(arc_process.get_value_from_key(key), new_value);

            Ok(())
        },
    );
}

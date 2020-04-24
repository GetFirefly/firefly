use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::Atom;

use crate::erlang::erase_1::result;
use crate::test::strategy;

#[test]
fn without_key_returns_undefined() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, key)| {
            prop_assert_eq!(result(&arc_process, key), Atom::str_to_term("undefined"));

            Ok(())
        },
    );
}

#[test]
fn with_key_returns_value_and_removes_key_from_dictionary() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, key, value)| {
            arc_process.put(key, value).unwrap();

            prop_assert_eq!(arc_process.get_value_from_key(key), value);

            prop_assert_eq!(result(&arc_process, key), value);

            prop_assert_eq!(
                arc_process.get_value_from_key(key),
                Atom::str_to_term("undefined")
            );

            Ok(())
        },
    );
}

use proptest::prop_assert_eq;
use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::Atom;

use crate::erlang::get_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_key_returns_undefined() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |key| {
                prop_assert_eq!(result(&arc_process, key), Atom::str_to_term("undefined"));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_key_returns_value() {
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

            Ok(())
        },
    );
}

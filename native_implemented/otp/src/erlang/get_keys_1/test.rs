mod with_entries;

use proptest::prop_assert_eq;
use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::get_keys_1::result;
use crate::test::strategy;

#[test]
fn without_entries_returns_empty_list() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, value)| {
            prop_assert_eq!(result(&arc_process, value), Ok(Term::NIL));

            Ok(())
        },
    );
}

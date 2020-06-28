mod with_empty_list;
mod with_non_empty_proper_list;

use std::convert::TryInto;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::concatenate_2::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_list_left_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, list, term)| {
            prop_assert_badarg!(
                result(&arc_process, list, term),
                format!("list ({}) is not a list", list)
            );

            Ok(())
        },
    );
}

#[test]
fn with_improper_list_left_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::list::improper(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, list, term)| {
            prop_assert_badarg!(
                result(&arc_process, list, term),
                format!("list ({}) is improper", list)
            );

            Ok(())
        },
    );
}

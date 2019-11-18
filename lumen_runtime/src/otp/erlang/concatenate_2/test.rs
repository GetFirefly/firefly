mod with_empty_list;
mod with_non_empty_proper_list;

use std::convert::TryInto;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::concatenate_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_list_left_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_list(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(&arc_process, left, right,), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_improper_list_left_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::list::improper(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(&arc_process, left, right,), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

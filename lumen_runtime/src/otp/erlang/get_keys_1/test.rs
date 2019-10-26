mod with_entries;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::Term;

use crate::otp::erlang::get_keys_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_entries_returns_empty_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |value| {
                prop_assert_eq!(native(&arc_process, value), Ok(Term::NIL));

                Ok(())
            })
            .unwrap();
    });
}

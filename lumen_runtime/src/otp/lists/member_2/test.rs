mod with_improper_list;
mod with_proper_non_empty_list;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::lists::member_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn with_empty_list_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |element| {
                let list = Term::NIL;

                prop_assert_eq!(native(element, list), Ok(false.into()));

                Ok(())
            })
            .unwrap();
    });
}

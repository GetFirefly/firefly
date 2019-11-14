mod with_float;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::float_to_list_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_float_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_float(arc_process.clone()),
                |float| {
                    prop_assert_eq!(native(&arc_process, float), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

mod with_list;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::maps::from_list_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::is_not_list(arc_process.clone())),
                |list| {
                    prop_assert_eq!(native(&arc_process, list), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

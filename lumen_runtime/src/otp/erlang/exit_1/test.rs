use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::exit;

use crate::otp::erlang::exit_1;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn exits_with_reason() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |reason| {
                prop_assert_eq!(exit_1::native(reason), Err(exit!(reason).into()));

                Ok(())
            })
            .unwrap();
    });
}

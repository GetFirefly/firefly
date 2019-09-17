use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::is_float_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_float_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_float(arc_process.clone()), |term| {
                prop_assert_eq!(native(term), false.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_float_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |term| {
                prop_assert_eq!(native(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::is_map_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_map_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_map(arc_process.clone()), |term| {
                prop_assert_eq!(native(term), false.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_map_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| strategy::term::is_map(arc_process.clone())),
            |term| {
                prop_assert_eq!(native(term), true.into());

                Ok(())
            },
        )
        .unwrap();
}

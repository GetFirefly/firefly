mod with_atom;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::is_boolean_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_boolean_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |term| {
                    prop_assert_eq!(native(term), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_boolean_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::is_boolean(), |term| {
            prop_assert_eq!(native(term), true.into());

            Ok(())
        })
        .unwrap();
}

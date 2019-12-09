use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::not_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_boolean_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |boolean| {
                    prop_assert_badarg!(
                        native(boolean),
                        format!("boolean ({}) must be a boolean", boolean)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_returns_true() {
    assert_eq!(native(false.into()), Ok(true.into()));
}

#[test]
fn with_true_returns_false() {
    assert_eq!(native(true.into()), Ok(false.into()));
}

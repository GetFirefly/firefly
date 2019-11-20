use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::erlang::not_1::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_boolean_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |boolean| {
                    prop_assert_eq!(
                        native(&arc_process, boolean),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_returns_true() {
    with_process(|process| {
        assert_eq!(native(process, false.into()), Ok(true.into()));
    });
}

#[test]
fn with_true_returns_false() {
    with_process(|process| {
        assert_eq!(native(process, true.into()), Ok(false.into()));
    });
}

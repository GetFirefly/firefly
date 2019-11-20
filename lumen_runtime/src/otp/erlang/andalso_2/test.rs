use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::erlang::andalso_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_boolean_left_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_boolean(arc_process.clone()),
                    strategy::term::is_boolean(),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        native(&arc_process, left, right),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_left_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |right| {
                prop_assert_eq!(native(&arc_process, false.into(), right), Ok(false.into()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_true_left_returns_right() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |right| {
                prop_assert_eq!(native(&arc_process, true.into(), right), Ok(right));

                Ok(())
            })
            .unwrap();
    });
}

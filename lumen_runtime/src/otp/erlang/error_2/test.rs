use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::error;

use crate::otp::erlang::error_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn errors_with_reason_and_arguments() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(reason, arguments)| {
                    prop_assert_eq!(
                        native(&arc_process, reason, arguments),
                        Err(error!(&arc_process, reason, arguments).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

use proptest::test_runner::{Config, TestCaseError, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::exception::{Exception, RuntimeException};

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
                    if let Err(Exception::Runtime(RuntimeException::Error(ref error))) =
                        native(reason, arguments)
                    {
                        prop_assert_eq!(error.reason(), reason);
                        prop_assert_eq!(error.arguments(), Some(arguments));

                        let source_string = format!("{:?}", error.source());
                        let expected_substring = "explicit error from Erlang";

                        prop_assert!(
                            source_string.contains(expected_substring),
                            "source ({}) does not contain `{}`",
                            source_string,
                            expected_substring
                        );

                        Ok(())
                    } else {
                        Err(TestCaseError::fail(format!("not an error")))
                    }
                },
            )
            .unwrap();
    });
}

mod with_false_left;
mod with_true_left;

use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::xor_2::native;
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
                    prop_assert_badarg!(
                        native(left, right),
                        format!("left ({}) must be a bool", left)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

mod with_proper_list_minuend;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::subtract_list_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_proper_list_minuend_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_proper_list(arc_process.clone()),
                    strategy::term::is_list(arc_process.clone()),
                ),
                |(minuend, subtrahend)| {
                    prop_assert_badarg!(
                        native(&arc_process, minuend, subtrahend),
                        format!("minuend ({}) is not a proper list", minuend)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

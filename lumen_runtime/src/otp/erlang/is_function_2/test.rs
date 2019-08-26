mod with_function;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::erlang::is_function_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_function_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_function(arc_process.clone()),
                    strategy::term::function::arity(arc_process.clone()),
                ),
                |(function, arity)| {
                    prop_assert_eq!(native(function, arity), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

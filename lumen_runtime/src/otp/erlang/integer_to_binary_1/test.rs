mod with_integer;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::integer_to_binary_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_integer_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_integer(arc_process.clone()),
                |integer| {
                    prop_assert_badarg!(
                        native(&arc_process, integer),
                        format!("integer ({}) is not an integer", integer)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

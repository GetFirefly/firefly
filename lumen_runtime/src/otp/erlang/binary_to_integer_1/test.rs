mod with_binary;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;

use crate::otp::erlang::binary_to_integer_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_binary(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        native(&arc_process, binary),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

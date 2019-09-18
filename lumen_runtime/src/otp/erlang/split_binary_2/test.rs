mod with_bitstring_binary;

use proptest::prop_assert_eq;
use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::erlang::split_binary_2::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_bitstring_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_bitstring(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(binary, position)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, position),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

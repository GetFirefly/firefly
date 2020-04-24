mod with_bitstring_binary;

use proptest::prop_assert_eq;
use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};

use crate::erlang::split_binary_2::result;
use crate::test::strategy;
use crate::test::{with_process, with_process_arc};

#[test]
fn without_bitstring_binary_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_bitstring(arc_process.clone()),
                strategy::term::integer::non_negative(arc_process.clone()),
            )
        },
        |(arc_process, binary, position)| {
            prop_assert_badarg!(
                result(&arc_process, binary, position),
                format!("binary ({}) is not a bitstring", binary)
            );

            Ok(())
        },
    );
}

mod with_binary;

use proptest::strategy::{Just, Strategy};
use proptest::{prop_assert, prop_assert_eq};

use crate::otp::erlang::binary_to_integer_1::native;
use crate::test::{run, strategy};

#[test]
fn without_binary_errors_badarg() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_binary(arc_process.clone()),
            )
        },
        |(arc_process, binary)| {
            prop_assert_badarg!(
                native(&arc_process, binary),
                format!("binary ({}) must be a binary", binary)
            );

            Ok(())
        },
    );
}

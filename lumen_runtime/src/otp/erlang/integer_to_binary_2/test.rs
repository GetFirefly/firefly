mod with_integer_integer;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::integer_to_binary_2::native;
use crate::test::{run, strategy};

#[test]
fn without_integer_integer_errors_badarg() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
                strategy::term::is_base(arc_process.clone()),
            )
        },
        |(arc_process, integer, base)| {
            prop_assert_badarg!(
                native(&arc_process, integer, base),
                format!("integer ({}) is not an integer", integer)
            );

            Ok(())
        },
    );
}

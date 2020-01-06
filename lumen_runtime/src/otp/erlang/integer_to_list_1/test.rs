mod with_integer;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::integer_to_list_1::native;
use crate::test::strategy;

#[test]
fn without_integer_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, integer)| {
            prop_assert_badarg!(
                native(&arc_process, integer),
                format!("integer ({}) is not an integer", integer)
            );

            Ok(())
        },
    );
}

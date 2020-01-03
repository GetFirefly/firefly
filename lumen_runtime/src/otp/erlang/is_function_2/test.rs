mod with_function;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;

use crate::otp::erlang::is_function_2::native;
use crate::test::{run, strategy};

#[test]
fn without_function_returns_false() {
    run(
        file!(),
        |arc_process| {
            (
                strategy::term::is_not_function(arc_process.clone()),
                strategy::term::function::arity(arc_process.clone()),
            )
        },
        |(function, arity)| {
            prop_assert_eq!(native(function, arity), Ok(false.into()));

            Ok(())
        },
    );
}

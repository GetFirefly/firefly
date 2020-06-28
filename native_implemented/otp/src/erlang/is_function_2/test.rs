mod with_function;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;

use crate::erlang::is_function_2::result;
use crate::test::strategy;

#[test]
fn without_function_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::is_not_function(arc_process.clone()),
                strategy::term::function::arity(arc_process.clone()),
            )
        },
        |(function, arity)| {
            prop_assert_eq!(result(function, arity), Ok(false.into()));

            Ok(())
        },
    );
}

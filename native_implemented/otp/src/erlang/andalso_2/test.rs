use proptest::prop_assert_eq;

use crate::erlang::andalso_2::result;
use crate::test::strategy;

#[test]
fn without_boolean_left_errors_badarg() {
    run!(
        |arc_process| {
            (
                strategy::term::is_not_boolean(arc_process.clone()),
                strategy::term::is_boolean(),
            )
        },
        |(left, right)| {
            prop_assert_badarg!(result(left, right), "left must be a bool");

            Ok(())
        },
    );
}

#[test]
fn with_false_left_returns_false() {
    run!(|arc_process| strategy::term(arc_process.clone()), |right| {
        prop_assert_eq!(result(false.into(), right), Ok(false.into()));

        Ok(())
    },);
}

#[test]
fn with_true_left_returns_right() {
    run!(|arc_process| strategy::term(arc_process.clone()), |right| {
        prop_assert_eq!(result(true.into(), right), Ok(right));

        Ok(())
    },);
}

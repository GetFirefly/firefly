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

// `with_false_left_returns_false` in integration tests

// `with_true_left_returns_right` in integration tests

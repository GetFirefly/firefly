use super::*;

#[test]
fn without_boolean_right_errors_badarg() {
    run!(
        |arc_process| strategy::term::is_not_boolean(arc_process.clone()),
        |right_boolean| {
            prop_assert_is_not_boolean!(result(false.into(), right_boolean), right_boolean);

            Ok(())
        },
    );
}

// `with_false_right_returns_false` in integration tests

// `with_true_right_returns_true` in integration tests

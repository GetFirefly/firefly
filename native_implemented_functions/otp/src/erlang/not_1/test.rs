use crate::erlang::not_1::result;
use crate::test::strategy;

#[test]
fn without_boolean_errors_badarg() {
    run!(
        |arc_process| strategy::term::is_not_boolean(arc_process.clone()),
        |boolean| {
            prop_assert_is_not_boolean!(result(boolean), boolean);

            Ok(())
        },
    );
}

#[test]
fn with_false_returns_true() {
    assert_eq!(result(false.into()), Ok(true.into()));
}

#[test]
fn with_true_returns_false() {
    assert_eq!(result(true.into()), Ok(false.into()));
}

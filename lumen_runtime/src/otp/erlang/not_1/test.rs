use crate::otp::erlang::not_1::native;
use crate::test::strategy;

#[test]
fn without_boolean_errors_badarg() {
    run!(
        |arc_process| strategy::term::is_not_boolean(arc_process.clone()),
        |boolean| {
            prop_assert_is_not_boolean!(native(boolean), boolean);

            Ok(())
        },
    );
}

#[test]
fn with_false_returns_true() {
    assert_eq!(native(false.into()), Ok(true.into()));
}

#[test]
fn with_true_returns_false() {
    assert_eq!(native(true.into()), Ok(false.into()));
}

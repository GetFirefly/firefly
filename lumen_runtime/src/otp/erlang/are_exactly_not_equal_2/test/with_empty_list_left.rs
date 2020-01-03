use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_empty_list_returns_true() {
    run(
        file!(),
        |arc_process| {
            (
                Just(Term::NIL),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be empty list", |v| !v.is_nil()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(native(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_right_returns_false() {
    assert_eq!(native(Term::NIL, Term::NIL), false.into());
}

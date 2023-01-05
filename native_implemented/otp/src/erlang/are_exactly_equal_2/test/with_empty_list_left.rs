use super::*;

#[test]
fn without_empty_list_returns_false() {
    run!(
        |arc_process| {
            (
                Just(Term::Nil),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be empty list", |v| !v.is_nil()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_right_returns_true() {
    assert_eq!(result(Term::Nil, Term::Nil), true.into());
}

use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::atom(),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_greater_atom_returns_true() {
    is_greater_than_or_equal(|_, _| Atom::str_to_term("keft"), true);
}

#[test]
fn with_same_atom_value_returns_true() {
    is_greater_than_or_equal(|_, _| Atom::str_to_term("left"), true);
}

#[test]
fn with_greater_atom_returns_false() {
    is_greater_than_or_equal(|_, _| Atom::str_to_term("meft"), false);
}

#[test]
fn without_number_or_atom_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::atom(),
                strategy::term(arc_process.clone())
                    .prop_filter("Right cannot be a number or atom", |right| {
                        !(right.is_atom() || right.is_number())
                    }),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

fn is_greater_than_or_equal<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_greater_than_or_equal(|_| Atom::str_to_term("left"), right, expected);
}

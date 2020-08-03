use super::*;

#[test]
fn with_number_second_returns_second() {
    run!(
        |arc_process| {
            (
                strategy::term::atom(),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), second);

            Ok(())
        },
    );
}

#[test]
fn with_lesser_atom_returns_second() {
    min(|_, _| Atom::str_to_term("eirst"), Second);
}

#[test]
fn with_same_atom_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_atom_value_returns_first() {
    min(|_, _| Atom::str_to_term("first"), First);
}

#[test]
fn with_greater_atom_returns_first() {
    min(|_, _| Atom::str_to_term("second"), First);
}

#[test]
fn without_number_or_atom_returns_first() {
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
        |(first, second)| {
            prop_assert_eq!(result(first, second), first);

            Ok(())
        },
    );
}

fn min<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::min(|_| Atom::str_to_term("first"), second, which);
}

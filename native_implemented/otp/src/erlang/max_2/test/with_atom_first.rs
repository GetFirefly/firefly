use super::*;

#[test]
fn with_number_second_returns_first() {
    run!(
        |arc_process| {
            (
                strategy::term::atom(),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), first);

            Ok(())
        },
    );
}

#[test]
fn with_lesser_atom_returns_first() {
    max(|_, _| Atom::str_to_term("eirst"), First);
}

#[test]
fn with_same_atom_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_atom_value_returns_first() {
    max(|_, _| Atom::str_to_term("first"), First);
}

#[test]
fn with_greater_atom_returns_second() {
    max(|_, _| Atom::str_to_term("second"), Second);
}

#[test]
fn without_number_or_atom_returns_second() {
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
            prop_assert_eq!(result(first, second), second);

            Ok(())
        },
    );
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::max(|_| Atom::str_to_term("first"), second, which);
}

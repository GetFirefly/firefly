use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_list_or_bitstring_second_returns_second() {
    run!(
        |arc_process| {
            (
                strategy::term::is_list(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("second cannot be a list or bitstring", |second| {
                        !(second.is_list() || second.is_bitstring())
                    }),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), second);

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_second_returns_second() {
    min(|_, _| Term::NIL, Second);
}

#[test]
fn with_lesser_list_second_returns_second() {
    min(
        |_, process| {
            process
                .cons(process.integer(0).unwrap(), process.integer(0).unwrap())
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_same_list_second_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_list_second_returns_first() {
    min(
        |_, process| {
            process
                .cons(process.integer(0).unwrap(), process.integer(1).unwrap())
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_greater_list_second_returns_first() {
    min(
        |_, process| {
            process
                .cons(process.integer(0).unwrap(), process.integer(2).unwrap())
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_bitstring_second_returns_first() {
    run!(
        |arc_process| {
            (
                strategy::term::is_list(arc_process.clone()),
                strategy::term::is_bitstring(arc_process.clone()),
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
    super::min(
        |process| {
            process
                .cons(process.integer(0).unwrap(), process.integer(1).unwrap())
                .unwrap()
        },
        second,
        which,
    );
}

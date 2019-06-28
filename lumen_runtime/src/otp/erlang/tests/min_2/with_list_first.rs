use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_list_or_bitstring_second_returns_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_list(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("second cannot be a list or bitstring", |second| {
                            !(second.is_list() || second.is_bitstring())
                        }),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::min_2(first, second), second);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_second_returns_second() {
    min(|_, _| Term::EMPTY_LIST, Second);
}

#[test]
fn with_lesser_list_second_returns_second() {
    min(
        |_, process| Term::cons(0.into_process(&process), 0.into_process(&process), &process),
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
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        First,
    );
}

#[test]
fn with_greater_list_second_returns_first() {
    min(
        |_, process| Term::cons(0.into_process(&process), 2.into_process(&process), &process),
        First,
    );
}

#[test]
fn with_bitstring_second_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_list(arc_process.clone()),
                    strategy::term::is_bitstring(arc_process.clone()),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::min_2(first, second), first);

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn min<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::min(
        |process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        second,
        which,
    );
}

use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_list_or_bitstring_second_returns_first() {
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
                    prop_assert_eq!(erlang::max_2(first, second), first.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_second_returns_first() {
    max(|_, _| Term::NIL, First);
}

#[test]
fn with_lesser_list_second_returns_first() {
    max(
        |_, process| {
            process
                .cons(process.integer(0).unwrap(), process.integer(0).unwrap())
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_list_second_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_list_second_returns_first() {
    max(
        |_, process| {
            process
                .cons(process.integer(0).unwrap(), process.integer(1).unwrap())
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_greater_list_second_returns_second() {
    max(
        |_, process| {
            process
                .cons(process.integer(0).unwrap(), process.integer(2).unwrap())
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_bitstring_second_returns_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_list(arc_process.clone()),
                    strategy::term::is_bitstring(arc_process.clone()),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::max_2(first, second), second);

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &ProcessControlBlock) -> Term,
{
    super::max(
        |process| {
            process
                .cons(process.integer(0).unwrap(), process.integer(1).unwrap())
                .unwrap()
        },
        second,
        which,
    );
}

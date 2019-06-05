use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_list_or_bitstring_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_list(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right cannot be a list or bitstring", |right| {
                            !(right.is_list() || right.is_bitstring())
                        }),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_greater_than_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_right_returns_true() {
    is_greater_than(|_, _| Term::EMPTY_LIST, true);
}

#[test]
fn with_greater_list_right_returns_true() {
    is_greater_than(
        |_, process| Term::cons(0.into_process(&process), 0.into_process(&process), &process),
        true,
    );
}

#[test]
fn with_same_list_right_returns_false() {
    is_greater_than(|left, _| left, false);
}

#[test]
fn with_same_value_list_right_returns_false() {
    is_greater_than(
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        false,
    );
}

#[test]
fn with_greater_list_right_returns_false() {
    is_greater_than(
        |_, process| Term::cons(0.into_process(&process), 2.into_process(&process), &process),
        false,
    );
}

#[test]
fn with_bitstring_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_list(arc_process.clone()),
                    strategy::term::is_bitstring(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_greater_than_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_greater_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_greater_than(
        |process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        right,
        expected,
    );
}

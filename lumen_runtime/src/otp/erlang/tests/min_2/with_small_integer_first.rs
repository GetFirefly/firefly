use super::*;

#[test]
fn with_lesser_small_integer_second_returns_second() {
    min(|_, process| (-1).into_process(&process), Second);
}

#[test]
fn with_same_small_integer_second_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_small_integer_second_returns_first() {
    min(|_, process| 0.into_process(&process), First);
}

#[test]
fn with_greater_small_integer_second_returns_first() {
    min(|_, process| 1.into_process(&process), First);
}

#[test]
fn with_lesser_big_integer_second_returns_second() {
    min(
        |_, process| (crate::integer::small::MIN - 1).into_process(&process),
        Second,
    )
}

#[test]
fn with_greater_big_integer_second_returns_first() {
    min(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        First,
    )
}

#[test]
fn with_lesser_float_second_returns_second() {
    min(|_, process| (-1.0).into_process(&process), Second)
}

#[test]
fn with_same_value_float_second_returns_first() {
    min(|_, process| 0.0.into_process(&process), First)
}

#[test]
fn with_greater_float_second_returns_first() {
    min(|_, process| 1.0.into_process(&process), First)
}

#[test]
fn without_number_second_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
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
    super::min(|process| 0.into_process(&process), second, which);
}

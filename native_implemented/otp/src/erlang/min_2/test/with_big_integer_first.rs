use super::*;

#[test]
fn with_lesser_small_integer_second_returns_second() {
    min(|_, process| process.integer(0).unwrap(), Second)
}

#[test]
fn with_greater_small_integer_second_returns_first() {
    super::min(
        |process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        |_, process| process.integer(SmallInteger::MIN_VALUE).unwrap(),
        First,
    );
}

#[test]
fn with_lesser_big_integer_second_returns_second() {
    min(
        |_, process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        Second,
    )
}

#[test]
fn with_same_big_integer_second_returns_first() {
    min(|first, _| first, First)
}

#[test]
fn with_same_value_big_integer_second_returns_first() {
    min(
        |_, process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
        First,
    )
}

#[test]
fn with_greater_big_integer_second_returns_first() {
    min(
        |_, process| process.integer(SmallInteger::MAX_VALUE + 2).unwrap(),
        First,
    )
}

#[test]
fn with_lesser_float_second_returns_second() {
    min(|_, process| process.float(1.0).unwrap(), Second)
}

#[test]
fn with_greater_float_second_returns_first() {
    super::min(
        |process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        |_, process| process.float(1.0).unwrap(),
        First,
    );
}

#[test]
fn without_second_number_returns_first() {
    run!(
        |arc_process| {
            (
                strategy::term::integer::big(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
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
        |process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
        second,
        which,
    );
}

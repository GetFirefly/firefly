use super::*;

#[test]
fn with_lesser_small_integer_second_returns_first() {
    max(|_, process| process.integer(0).unwrap(), First)
}

#[test]
fn with_greater_small_integer_second_returns_second() {
    super::max(
        |process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        |_, process| process.integer(SmallInteger::MIN_VALUE).unwrap(),
        Second,
    );
}

#[test]
fn with_lesser_big_integer_second_returns_first() {
    max(
        |_, process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        First,
    )
}

#[test]
fn with_same_big_integer_second_returns_first() {
    max(|first, _| first, First)
}

#[test]
fn with_same_value_big_integer_second_returns_first() {
    max(
        |_, process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
        First,
    )
}

#[test]
fn with_greater_big_integer_second_returns_second() {
    max(
        |_, process| process.integer(SmallInteger::MAX_VALUE + 2).unwrap(),
        Second,
    )
}

#[test]
fn with_lesser_float_second_returns_first() {
    max(|_, process| process.float(1.0).unwrap(), First)
}

#[test]
fn with_greater_float_second_returns_second() {
    super::max(
        |process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        |_, process| process.float(1.0).unwrap(),
        Second,
    );
}

#[test]
fn without_second_number_returns_second() {
    run!(
        |arc_process| {
            (
                strategy::term::integer::big(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
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
    super::max(
        |process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
        second,
        which,
    );
}

use super::*;

#[test]
fn with_greater_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, process| process.integer(-1), true)
}

#[test]
fn with_equal_small_integer_right_returns_false() {
    is_greater_than_or_equal(|_, process| process.integer(1), true)
}

#[test]
fn with_greater_small_integer_right_returns_false() {
    is_greater_than_or_equal(|_, process| process.integer(2), false)
}

#[test]
fn with_lesser_big_integer_right_returns_true() {
    is_greater_than_or_equal(
        |_, process| process.integer(SmallInteger::MIN_VALUE - 1),
        true,
    )
}

#[test]
fn with_greater_big_integer_right_returns_false() {
    is_greater_than_or_equal(
        |_, process| process.integer(SmallInteger::MAX_VALUE + 1),
        false,
    )
}

#[test]
fn with_greater_float_right_returns_true() {
    is_greater_than_or_equal(|_, process| process.float(-1.0), true)
}

#[test]
fn with_greater_float_right_returns_false() {
    is_greater_than_or_equal(|_, process| process.float(2.0), false)
}

#[test]
fn without_number_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::float(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
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
    super::is_greater_than_or_equal(|process| process.float(1.0), right, expected);
}

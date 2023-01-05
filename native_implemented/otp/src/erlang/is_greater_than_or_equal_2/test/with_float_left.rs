use super::*;

#[test]
fn with_greater_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, process| process.integer(-1).unwrap(), true)
}

#[test]
fn with_equal_small_integer_right_returns_false() {
    is_greater_than_or_equal(|_, process| process.integer(1).unwrap(), true)
}

#[test]
fn with_greater_small_integer_right_returns_false() {
    is_greater_than_or_equal(|_, process| process.integer(2).unwrap(), false)
}

#[test]
fn with_lesser_big_integer_right_returns_true() {
    is_greater_than_or_equal(
        |_, process| process.integer(Integer::MIN_SMALL - 1).unwrap(),
        true,
    )
}

#[test]
fn with_greater_big_integer_right_returns_false() {
    is_greater_than_or_equal(
        |_, process| process.integer(Integer::MAX_SMALL + 1).unwrap(),
        false,
    )
}

#[test]
fn with_greater_float_right_returns_true() {
    is_greater_than_or_equal(|_, process| -1.0.into(), true)
}

#[test]
fn with_greater_float_right_returns_false() {
    is_greater_than_or_equal(|_, process| 2.0.into(), false)
}

#[test]
fn without_number_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::float(),
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
    super::is_greater_than_or_equal(|process| 1.0.into(), right, expected);
}

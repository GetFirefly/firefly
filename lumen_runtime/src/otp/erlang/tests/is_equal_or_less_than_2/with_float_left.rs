use super::*;

#[test]
fn with_lesser_small_integer_right_returns_false() {
    is_equal_or_less_than(|_, process| process.integer(-1).unwrap(), false)
}

#[test]
fn with_equal_small_integer_right_returns_true() {
    is_equal_or_less_than(|_, process| process.integer(1).unwrap(), true)
}

#[test]
fn with_greater_small_integer_right_returns_true() {
    is_equal_or_less_than(|_, process| process.integer(2).unwrap(), true)
}

#[test]
fn with_lesser_big_integer_right_returns_false() {
    is_equal_or_less_than(
        |_, process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        false,
    )
}

#[test]
fn with_greater_big_integer_right_returns_true() {
    is_equal_or_less_than(
        |_, process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
        true,
    )
}

#[test]
fn with_lesser_float_right_returns_false() {
    is_equal_or_less_than(|_, process| process.float(-1.0).unwrap(), false)
}

#[test]
fn with_equal_float_right_returns_true() {
    is_equal_or_less_than(|_, process| process.float(1.0).unwrap(), true)
}

#[test]
fn with_greater_float_right_returns_true() {
    is_equal_or_less_than(|_, process| process.float(2.0).unwrap(), true)
}

#[test]
fn without_number_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_equal_or_less_than_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_equal_or_less_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &ProcessControlBlock) -> Term,
{
    super::is_equal_or_less_than(|process| process.float(1.0).unwrap(), right, expected);
}

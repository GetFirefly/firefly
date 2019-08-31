use super::*;

#[test]
fn with_greater_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, process| process.integer(0).unwrap(), true)
}

#[test]
fn with_greater_small_integer_right_returns_false() {
    super::is_greater_than_or_equal(
        |process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        |_, process| process.integer(SmallInteger::MIN_VALUE).unwrap(),
        false,
    );
}

#[test]
fn with_greater_big_integer_right_returns_true() {
    is_greater_than_or_equal(
        |_, process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        true,
    )
}

#[test]
fn with_same_value_big_integer_right_returns_true() {
    is_greater_than_or_equal(
        |_, process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
        true,
    )
}

#[test]
fn with_greater_big_integer_right_returns_false() {
    is_greater_than_or_equal(
        |_, process| process.integer(SmallInteger::MAX_VALUE + 2).unwrap(),
        false,
    )
}

#[test]
fn with_greater_float_right_returns_true() {
    is_greater_than_or_equal(|_, process| process.float(1.0).unwrap(), true)
}

#[test]
fn with_greater_float_right_returns_false() {
    super::is_greater_than_or_equal(
        |process| process.integer(SmallInteger::MIN_VALUE - 1).unwrap(),
        |_, process| process.float(1.0).unwrap(),
        false,
    );
}

#[test]
fn without_number_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::is_greater_than_or_equal_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_greater_than_or_equal<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_greater_than_or_equal(
        |process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap(),
        right,
        expected,
    );
}

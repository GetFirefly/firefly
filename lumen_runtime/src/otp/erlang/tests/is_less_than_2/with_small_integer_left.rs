use super::*;

#[test]
fn with_lesser_small_integer_right_returns_false() {
    is_less_than(|_, process| (-1).into_process(&process), false);
}

#[test]
fn with_same_small_integer_right_returns_false() {
    is_less_than(|left, _| left, false);
}

#[test]
fn with_same_value_small_integer_right_returns_false() {
    is_less_than(|_, process| 0.into_process(&process), false);
}

#[test]
fn with_greater_small_integer_right_returns_true() {
    is_less_than(|_, process| 1.into_process(&process), true);
}

#[test]
fn with_lesser_big_integer_right_returns_false() {
    is_less_than(
        |_, process| (crate::integer::small::MIN - 1).into_process(&process),
        false,
    )
}

#[test]
fn with_greater_big_integer_right_returns_true() {
    is_less_than(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        true,
    )
}

#[test]
fn with_lesser_float_right_returns_false() {
    is_less_than(|_, process| (-1.0).into_process(&process), false)
}

#[test]
fn with_same_value_float_right_returns_false() {
    is_less_than(|_, process| 0.0.into_process(&process), false)
}

#[test]
fn with_greater_float_right_returns_true() {
    is_less_than(|_, process| 1.0.into_process(&process), true)
}

#[test]
fn without_number_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_less_than_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_less_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_less_than(|process| 0.into_process(&process), right, expected);
}

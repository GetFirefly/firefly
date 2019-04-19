use super::*;

#[test]
fn with_greater_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, process| (-1).into_process(&process), true);
}

#[test]
fn with_same_small_integer_right_returns_true() {
    is_greater_than_or_equal(|left, _| left, true);
}

#[test]
fn with_same_value_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, process| 0.into_process(&process), true);
}

#[test]
fn with_greater_small_integer_right_returns_false() {
    is_greater_than_or_equal(|_, process| 1.into_process(&process), false);
}

#[test]
fn with_greater_big_integer_right_returns_true() {
    is_greater_than_or_equal(
        |_, process| (crate::integer::small::MIN - 1).into_process(&process),
        true,
    )
}

#[test]
fn with_greater_big_integer_right_returns_false() {
    is_greater_than_or_equal(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        false,
    )
}

#[test]
fn with_greater_float_right_returns_true() {
    is_greater_than_or_equal(|_, process| (-1.0).into_process(&process), true)
}

#[test]
fn with_same_value_float_right_returns_true() {
    is_greater_than_or_equal(|_, process| 0.0.into_process(&process), true)
}

#[test]
fn with_greater_float_right_returns_false() {
    is_greater_than_or_equal(|_, process| 1.0.into_process(&process), false)
}

#[test]
fn with_atom_right_returns_false() {
    is_greater_than_or_equal(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), false);
}

#[test]
fn with_local_reference_right_returns_false() {
    is_greater_than_or_equal(|_, process| Term::local_reference(&process), false);
}

#[test]
fn with_local_pid_right_returns_false() {
    is_greater_than_or_equal(|_, _| Term::local_pid(0, 1).unwrap(), false);
}

#[test]
fn with_external_pid_right_returns_false() {
    is_greater_than_or_equal(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        false,
    );
}

#[test]
fn with_tuple_right_returns_false() {
    is_greater_than_or_equal(|_, process| Term::slice_to_tuple(&[], &process), false);
}

#[test]
fn with_map_right_returns_false() {
    is_greater_than_or_equal(|_, process| Term::slice_to_map(&[], &process), false);
}

#[test]
fn with_empty_list_right_returns_false() {
    is_greater_than_or_equal(|_, _| Term::EMPTY_LIST, false);
}

#[test]
fn with_list_right_returns_false() {
    is_greater_than_or_equal(
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        false,
    );
}

#[test]
fn with_heap_binary_right_returns_false() {
    is_greater_than_or_equal(|_, process| Term::slice_to_binary(&[], &process), false);
}

#[test]
fn with_subbinary_right_returns_false() {
    is_greater_than_or_equal(|_, process| bitstring!(1 :: 1, &process), false);
}

fn is_greater_than_or_equal<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_greater_than_or_equal(|process| 0.into_process(&process), right, expected);
}

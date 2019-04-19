use super::*;

#[test]
fn with_atom_right_returns_false() {
    are_equal_after_conversion(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), false)
}

#[test]
fn with_local_reference_right_returns_false() {
    are_equal_after_conversion(|_, process| Term::local_reference(&process), false);
}

#[test]
fn with_empty_list_right_returns_false() {
    are_equal_after_conversion(|_, _| Term::EMPTY_LIST, false);
}

#[test]
fn with_list_right_returns_false() {
    are_equal_after_conversion(
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        false,
    );
}

#[test]
fn with_small_integer_right_returns_false() {
    are_equal_after_conversion(|_, process| 0.into_process(&process), false)
}

#[test]
fn with_big_integer_right_returns_false() {
    are_equal_after_conversion(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        false,
    )
}

#[test]
fn with_float_right_returns_false() {
    are_equal_after_conversion(|_, process| 0.0.into_process(&process), false)
}

#[test]
fn with_local_pid_right_returns_false() {
    are_equal_after_conversion(|_, _| Term::local_pid(2, 3).unwrap(), false);
}

#[test]
fn with_same_external_pid_right_returns_true() {
    are_equal_after_conversion(|left, _| left, true);
}

#[test]
fn with_same_value_external_pid_right_returns_true() {
    are_equal_after_conversion(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        true,
    );
}

#[test]
fn with_different_external_pid_right_returns_false() {
    are_equal_after_conversion(
        |_, process| Term::external_pid(4, 5, 6, &process).unwrap(),
        false,
    );
}

#[test]
fn with_tuple_right_returns_false() {
    are_equal_after_conversion(|_, process| Term::slice_to_tuple(&[], &process), false);
}

#[test]
fn with_map_right_returns_false() {
    are_equal_after_conversion(|_, process| Term::slice_to_map(&[], &process), false);
}

#[test]
fn with_heap_binary_right_returns_false() {
    are_equal_after_conversion(|_, process| Term::slice_to_binary(&[], &process), false);
}

#[test]
fn with_subbinary_right_returns_false() {
    are_equal_after_conversion(|_, process| bitstring!(1 :: 1, &process), false);
}

fn are_equal_after_conversion<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::are_equal_after_conversion(
        |process| Term::external_pid(1, 2, 3, &process).unwrap(),
        right,
        expected,
    );
}

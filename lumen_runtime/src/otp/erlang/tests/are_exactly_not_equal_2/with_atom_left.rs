use super::*;

#[test]
fn with_same_atom_returns_false() {
    are_exactly_not_equal(|left, _| left, false);
}

#[test]
fn with_same_atom_value_returns_false() {
    are_exactly_not_equal(|_, _| Term::str_to_atom("left", DoNotCare).unwrap(), false);
}

#[test]
fn with_different_atom_returns_true() {
    are_exactly_not_equal(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), true);
}

#[test]
fn with_local_reference_right_returns_true() {
    are_exactly_not_equal(|_, process| Term::next_local_reference(process), true);
}

#[test]
fn with_empty_list_right_returns_true() {
    are_exactly_not_equal(|_, _| Term::EMPTY_LIST, true);
}

#[test]
fn with_list_right_returns_true() {
    are_exactly_not_equal(
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        true,
    );
}

#[test]
fn with_small_integer_right_returns_true() {
    are_exactly_not_equal(|_, process| 0.into_process(&process), true)
}

#[test]
fn with_big_integer_right_returns_true() {
    are_exactly_not_equal(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        true,
    )
}

#[test]
fn with_float_right_returns_true() {
    are_exactly_not_equal(|_, process| 0.0.into_process(&process), true)
}

#[test]
fn with_local_pid_right_returns_true() {
    are_exactly_not_equal(|_, _| Term::local_pid(0, 1).unwrap(), true);
}

#[test]
fn with_external_pid_right_returns_true() {
    are_exactly_not_equal(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        true,
    );
}

#[test]
fn with_tuple_right_returns_true() {
    are_exactly_not_equal(|_, process| Term::slice_to_tuple(&[], &process), true);
}

#[test]
fn with_map_right_returns_true() {
    are_exactly_not_equal(|_, process| Term::slice_to_map(&[], &process), true);
}

#[test]
fn with_heap_binary_right_returns_true() {
    are_exactly_not_equal(|_, process| Term::slice_to_binary(&[], &process), true);
}

#[test]
fn with_subbinary_right_returns_true() {
    are_exactly_not_equal(|_, process| bitstring!(1 :: 1, &process), true);
}

fn are_exactly_not_equal<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::are_exactly_not_equal(
        |_| Term::str_to_atom("left", DoNotCare).unwrap(),
        right,
        expected,
    );
}

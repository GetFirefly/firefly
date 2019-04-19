use super::*;

#[test]
fn with_atom_right_returns_true() {
    are_exactly_not_equal(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), true)
}

#[test]
fn with_local_reference_right_returns_true() {
    are_exactly_not_equal(|_, process| Term::local_reference(&process), true);
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
    are_exactly_not_equal(|_, _| Term::local_pid(2, 3).unwrap(), true);
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
fn with_same_map_right_returns_false() {
    are_exactly_not_equal(|left, _| left, false);
}

#[test]
fn with_same_value_map_right_returns_false() {
    are_exactly_not_equal(|_, process| Term::slice_to_map(&[], &process), false);
}

#[test]
fn with_different_map_right_returns_true() {
    are_exactly_not_equal(
        |_, process| {
            Term::slice_to_map(
                &[(
                    Term::str_to_atom("a", DoNotCare).unwrap(),
                    1.into_process(&process),
                )],
                &process,
            )
        },
        true,
    );
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
    super::are_exactly_not_equal(|process| Term::slice_to_map(&[], &process), right, expected);
}

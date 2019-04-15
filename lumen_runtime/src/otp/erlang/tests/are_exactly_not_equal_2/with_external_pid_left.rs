use super::*;

#[test]
fn with_atom_right_returns_true() {
    are_exactly_not_equal(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), true)
}

#[test]
fn with_local_reference_right_returns_true() {
    are_exactly_not_equal(|_, mut process| Term::local_reference(&mut process), true);
}

#[test]
fn with_empty_list_right_returns_true() {
    are_exactly_not_equal(|_, _| Term::EMPTY_LIST, true);
}

#[test]
fn with_list_right_returns_true() {
    are_exactly_not_equal(
        |_, mut process| {
            Term::cons(
                0.into_process(&mut process),
                1.into_process(&mut process),
                &mut process,
            )
        },
        true,
    );
}

#[test]
fn with_small_integer_right_returns_true() {
    are_exactly_not_equal(|_, mut process| 0.into_process(&mut process), true)
}

#[test]
fn with_big_integer_right_returns_true() {
    are_exactly_not_equal(
        |_, mut process| (crate::integer::small::MAX + 1).into_process(&mut process),
        true,
    )
}

#[test]
fn with_float_right_returns_true() {
    are_exactly_not_equal(|_, mut process| 0.0.into_process(&mut process), true)
}

#[test]
fn with_local_pid_right_returns_true() {
    are_exactly_not_equal(|_, _| Term::local_pid(2, 3).unwrap(), true);
}

#[test]
fn with_same_external_pid_right_returns_false() {
    are_exactly_not_equal(|left, _| left, false);
}

#[test]
fn with_same_value_external_pid_right_returns_false() {
    are_exactly_not_equal(
        |_, mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        false,
    );
}

#[test]
fn with_different_external_pid_right_returns_true() {
    are_exactly_not_equal(
        |_, mut process| Term::external_pid(4, 5, 6, &mut process).unwrap(),
        true,
    );
}

#[test]
fn with_tuple_right_returns_true() {
    are_exactly_not_equal(
        |_, mut process| Term::slice_to_tuple(&[], &mut process),
        true,
    );
}

#[test]
fn with_map_right_returns_true() {
    are_exactly_not_equal(|_, mut process| Term::slice_to_map(&[], &mut process), true);
}

#[test]
fn with_heap_binary_right_returns_true() {
    are_exactly_not_equal(
        |_, mut process| Term::slice_to_binary(&[], &mut process),
        true,
    );
}

#[test]
fn with_subbinary_right_returns_true() {
    are_exactly_not_equal(|_, mut process| bitstring!(1 :: 1, &mut process), true);
}

fn are_exactly_not_equal<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &mut Process) -> Term,
{
    super::are_exactly_not_equal(
        |mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        right,
        expected,
    );
}

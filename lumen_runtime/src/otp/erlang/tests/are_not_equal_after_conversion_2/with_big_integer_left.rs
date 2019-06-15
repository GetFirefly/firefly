use super::*;

#[test]
fn with_atom_returns_true() {
    are_not_equal_after_conversion(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), true);
}

#[test]
fn with_local_reference_right_returns_true() {
    are_not_equal_after_conversion(|_, process| Term::next_local_reference(process), true);
}

#[test]
fn with_empty_list_right_returns_true() {
    are_not_equal_after_conversion(|_, _| Term::EMPTY_LIST, true);
}

#[test]
fn with_list_right_returns_true() {
    are_not_equal_after_conversion(
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        true,
    );
}

#[test]
fn with_small_integer_right_returns_true() {
    are_not_equal_after_conversion(|_, process| 0.into_process(&process), true)
}

#[test]
fn with_same_big_integer_right_returns_false() {
    are_not_equal_after_conversion(|left, _| left, false)
}

#[test]
fn with_same_value_big_integer_right_returns_false() {
    are_not_equal_after_conversion(
        |_, process| (crate::integer::small::MIN - 1).into_process(&process),
        false,
    )
}

#[test]
fn with_different_big_integer_right_returns_true() {
    are_not_equal_after_conversion(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        true,
    )
}

#[test]
fn with_same_value_float_right_returns_false() {
    let i = crate::integer::small::MIN - 1;
    let f = i as f64;

    // part of the big integer range can fit in an f64
    if crate::float::INTEGRAL_MIN < f {
        are_not_equal_after_conversion(|_, process| f.into_process(&process), false)
    }
}

#[test]
fn with_different_value_float_right_returns_true() {
    are_not_equal_after_conversion(|_, process| 1.0.into_process(&process), true)
}

#[test]
fn with_local_pid_right_returns_true() {
    are_not_equal_after_conversion(|_, _| Term::local_pid(0, 1).unwrap(), true);
}

#[test]
fn with_external_pid_right_returns_true() {
    are_not_equal_after_conversion(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        true,
    );
}

#[test]
fn with_tuple_right_returns_true() {
    are_not_equal_after_conversion(|_, process| Term::slice_to_tuple(&[], &process), true);
}

#[test]
fn with_map_right_returns_true() {
    are_not_equal_after_conversion(|_, process| Term::slice_to_map(&[], &process), true);
}

#[test]
fn with_heap_binary_right_returns_true() {
    are_not_equal_after_conversion(|_, process| Term::slice_to_binary(&[], &process), true);
}

#[test]
fn with_subbinary_right_returns_true() {
    are_not_equal_after_conversion(|_, process| bitstring!(1 :: 1, &process), true);
}

fn are_not_equal_after_conversion<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::are_not_equal_after_conversion(
        |process| (crate::integer::small::MIN - 1).into_process(&process),
        right,
        expected,
    );
}

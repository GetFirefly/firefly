use super::*;

#[test]
fn with_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, mut process| 0.into_process(&mut process), true)
}

#[test]
fn with_big_integer_right_returns_true() {
    is_greater_than_or_equal(
        |_, mut process| (crate::integer::small::MAX + 1).into_process(&mut process),
        true,
    )
}

#[test]
fn with_float_right_returns_true() {
    is_greater_than_or_equal(|_, mut process| 0.0.into_process(&mut process), true)
}

#[test]
fn with_atom_returns_true() {
    is_greater_than_or_equal(|_, _| Term::str_to_atom("meft", DoNotCare).unwrap(), true);
}

#[test]
fn with_greater_local_reference_right_returns_true() {
    is_greater_than_or_equal(
        |_, mut process| Term::number_to_local_reference(0, &mut process),
        true,
    );
}

#[test]
fn with_same_local_reference_right_returns_true() {
    is_greater_than_or_equal(|left, _| left, true);
}

#[test]
fn with_same_value_local_reference_right_returns_true() {
    is_greater_than_or_equal(
        |_, mut process| Term::number_to_local_reference(1, &mut process),
        true,
    );
}

#[test]
fn with_greater_local_reference_right_returns_false() {
    is_greater_than_or_equal(
        |_, mut process| Term::number_to_local_reference(2, &mut process),
        false,
    );
}

#[test]
fn with_local_pid_right_returns_false() {
    is_greater_than_or_equal(|_, _| Term::local_pid(0, 1).unwrap(), false);
}

#[test]
fn with_external_pid_right_returns_false() {
    is_greater_than_or_equal(
        |_, mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        false,
    );
}

#[test]
fn with_tuple_right_returns_false() {
    is_greater_than_or_equal(
        |_, mut process| Term::slice_to_tuple(&[], &mut process),
        false,
    );
}

#[test]
fn with_map_right_returns_false() {
    is_greater_than_or_equal(
        |_, mut process| Term::slice_to_map(&[], &mut process),
        false,
    );
}

#[test]
fn with_empty_list_right_returns_false() {
    is_greater_than_or_equal(|_, _| Term::EMPTY_LIST, false);
}

#[test]
fn with_list_right_returns_false() {
    is_greater_than_or_equal(
        |_, mut process| {
            Term::cons(
                0.into_process(&mut process),
                1.into_process(&mut process),
                &mut process,
            )
        },
        false,
    );
}

#[test]
fn with_heap_binary_right_returns_false() {
    is_greater_than_or_equal(
        |_, mut process| Term::slice_to_binary(&[], &mut process),
        false,
    );
}

#[test]
fn with_subbinary_right_returns_false() {
    is_greater_than_or_equal(|_, mut process| bitstring!(1 :: 1, &mut process), false);
}

fn is_greater_than_or_equal<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &mut Process) -> Term,
{
    super::is_greater_than_or_equal(
        |mut process| Term::number_to_local_reference(1, &mut process),
        right,
        expected,
    );
}

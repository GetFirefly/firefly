use super::*;

#[test]
fn with_greater_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, mut process| (-1).into_process(&mut process), true);
}

#[test]
fn with_same_small_integer_right_returns_true() {
    is_greater_than_or_equal(|left, _| left, true);
}

#[test]
fn with_same_value_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, mut process| 0.into_process(&mut process), true);
}

#[test]
fn with_greater_small_integer_right_returns_false() {
    is_greater_than_or_equal(|_, mut process| 1.into_process(&mut process), false);
}

#[test]
fn with_greater_big_integer_right_returns_true() {
    is_greater_than_or_equal(
        |_, mut process| (crate::integer::small::MIN - 1).into_process(&mut process),
        true,
    )
}

#[test]
fn with_greater_big_integer_right_returns_false() {
    is_greater_than_or_equal(
        |_, mut process| (crate::integer::small::MAX + 1).into_process(&mut process),
        false,
    )
}

#[test]
fn with_greater_float_right_returns_true() {
    is_greater_than_or_equal(|_, mut process| (-1.0).into_process(&mut process), true)
}

#[test]
fn with_same_value_float_right_returns_true() {
    is_greater_than_or_equal(|_, mut process| 0.0.into_process(&mut process), true)
}

#[test]
fn with_greater_float_right_returns_false() {
    is_greater_than_or_equal(|_, mut process| 1.0.into_process(&mut process), false)
}

#[test]
fn with_atom_right_returns_false() {
    is_greater_than_or_equal(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), false);
}

#[test]
fn with_local_reference_right_returns_false() {
    is_greater_than_or_equal(|_, mut process| Term::local_reference(&mut process), false);
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
    super::is_greater_than_or_equal(|mut process| 0.into_process(&mut process), right, expected);
}

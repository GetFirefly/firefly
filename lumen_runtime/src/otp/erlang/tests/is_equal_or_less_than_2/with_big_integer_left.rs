use super::*;

#[test]
fn with_lesser_small_integer_right_returns_false() {
    is_equal_or_less_than(|_, mut process| 0.into_process(&mut process), false)
}

#[test]
fn with_greater_small_integer_right_returns_true() {
    super::is_equal_or_less_than(
        |mut process| (crate::integer::small::MIN - 1).into_process(&mut process),
        |_, mut process| crate::integer::small::MIN.into_process(&mut process),
        true,
    );
}

#[test]
fn with_lesser_big_integer_right_returns_false() {
    is_equal_or_less_than(
        |_, mut process| (crate::integer::small::MIN - 1).into_process(&mut process),
        false,
    )
}

#[test]
fn with_same_big_integer_right_returns_true() {
    is_equal_or_less_than(|left, _| left, true)
}

#[test]
fn with_same_value_big_integer_right_returns_true() {
    is_equal_or_less_than(
        |_, mut process| (crate::integer::small::MAX + 1).into_process(&mut process),
        true,
    )
}

#[test]
fn with_greater_big_integer_right_returns_true() {
    is_equal_or_less_than(
        |_, mut process| (crate::integer::small::MAX + 2).into_process(&mut process),
        true,
    )
}

#[test]
fn with_lesser_float_right_returns_false() {
    is_equal_or_less_than(|_, mut process| 0.0.into_process(&mut process), false)
}

#[test]
fn with_greater_float_right_returns_true() {
    super::is_equal_or_less_than(
        |mut process| (crate::integer::small::MIN - 1).into_process(&mut process),
        |_, mut process| 0.0.into_process(&mut process),
        true,
    );
}

#[test]
fn with_atom_right_returns_true() {
    is_equal_or_less_than(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), true);
}

#[test]
fn with_local_reference_right_returns_true() {
    is_equal_or_less_than(|_, mut process| Term::local_reference(&mut process), true);
}

#[test]
fn with_local_pid_right_returns_true() {
    is_equal_or_less_than(|_, _| Term::local_pid(0, 1).unwrap(), true);
}

#[test]
fn with_external_pid_right_returns_true() {
    is_equal_or_less_than(
        |_, mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        true,
    );
}

#[test]
fn with_tuple_right_returns_true() {
    is_equal_or_less_than(
        |_, mut process| Term::slice_to_tuple(&[], &mut process),
        true,
    );
}

#[test]
fn with_map_right_returns_true() {
    is_equal_or_less_than(|_, mut process| Term::slice_to_map(&[], &mut process), true);
}

#[test]
fn with_empty_list_right_returns_true() {
    is_equal_or_less_than(|_, _| Term::EMPTY_LIST, true);
}

#[test]
fn with_list_right_returns_true() {
    is_equal_or_less_than(
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
fn with_heap_binary_right_returns_true() {
    is_equal_or_less_than(
        |_, mut process| Term::slice_to_binary(&[], &mut process),
        true,
    );
}

#[test]
fn with_subbinary_right_returns_true() {
    is_equal_or_less_than(|_, mut process| bitstring!(1 :: 1, &mut process), true);
}

fn is_equal_or_less_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &mut Process) -> Term,
{
    super::is_equal_or_less_than(
        |mut process| (crate::integer::small::MAX + 1).into_process(&mut process),
        right,
        expected,
    );
}

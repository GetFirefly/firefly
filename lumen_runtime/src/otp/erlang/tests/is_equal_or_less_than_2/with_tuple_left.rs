use super::*;

#[test]
fn with_small_integer_right_returns_false() {
    is_equal_or_less_than(|_, process| 0.into_process(&process), false)
}

#[test]
fn with_big_integer_right_returns_false() {
    is_equal_or_less_than(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        false,
    )
}

#[test]
fn with_float_right_returns_false() {
    is_equal_or_less_than(|_, process| 0.0.into_process(&process), false)
}

#[test]
fn with_atom_returns_false() {
    is_equal_or_less_than(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), false);
}

#[test]
fn with_local_reference_right_returns_false() {
    is_equal_or_less_than(|_, process| Term::next_local_reference(process), false);
}

#[test]
fn with_local_pid_right_returns_false() {
    is_equal_or_less_than(|_, _| Term::local_pid(0, 1).unwrap(), false);
}

#[test]
fn with_external_pid_right_returns_false() {
    is_equal_or_less_than(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        false,
    );
}

#[test]
fn with_smaller_tuple_right_returns_false() {
    is_equal_or_less_than(
        |_, process| Term::slice_to_tuple(&[1.into_process(&process)], &process),
        false,
    );
}

#[test]
fn with_same_size_tuple_with_lesser_elements_returns_false() {
    is_equal_or_less_than(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 1.into_process(&process)],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_same_tuple_returns_true() {
    is_equal_or_less_than(|left, _| left, true);
}

#[test]
fn with_same_value_tuple_returns_true() {
    is_equal_or_less_than(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 2.into_process(&process)],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_true() {
    is_equal_or_less_than(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 3.into_process(&process)],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_greater_size_tuple_returns_true() {
    is_equal_or_less_than(
        |_, process| {
            Term::slice_to_tuple(
                &[
                    1.into_process(&process),
                    2.into_process(&process),
                    3.into_process(&process),
                ],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_map_right_returns_true() {
    is_equal_or_less_than(|_, process| Term::slice_to_map(&[], &process), true);
}

#[test]
fn with_empty_list_right_returns_true() {
    is_equal_or_less_than(|_, _| Term::EMPTY_LIST, true);
}

#[test]
fn with_list_right_returns_true() {
    is_equal_or_less_than(
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        true,
    );
}

#[test]
fn with_heap_binary_right_returns_true() {
    is_equal_or_less_than(|_, process| Term::slice_to_binary(&[], &process), true);
}

#[test]
fn with_subbinary_right_returns_true() {
    is_equal_or_less_than(|_, process| bitstring!(1 :: 1, &process), true);
}

fn is_equal_or_less_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_equal_or_less_than(
        |process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 2.into_process(&process)],
                &process,
            )
        },
        right,
        expected,
    );
}

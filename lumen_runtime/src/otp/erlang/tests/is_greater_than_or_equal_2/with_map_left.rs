use super::*;

#[test]
fn with_small_integer_right_returns_true() {
    is_greater_than_or_equal(|_, process| 0.into_process(&process), true)
}

#[test]
fn with_big_integer_right_returns_true() {
    is_greater_than_or_equal(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        true,
    )
}

#[test]
fn with_float_right_returns_true() {
    is_greater_than_or_equal(|_, process| 0.0.into_process(&process), true)
}

#[test]
fn with_atom_returns_true() {
    is_greater_than_or_equal(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), true);
}

#[test]
fn with_local_reference_right_returns_true() {
    is_greater_than_or_equal(|_, process| Term::next_local_reference(process), true);
}

#[test]
fn with_local_pid_right_returns_true() {
    is_greater_than_or_equal(|_, _| Term::local_pid(0, 1).unwrap(), true);
}

#[test]
fn with_external_pid_right_returns_true() {
    is_greater_than_or_equal(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        true,
    );
}

#[test]
fn with_tuple_right_returns_true() {
    is_greater_than_or_equal(|_, process| Term::slice_to_tuple(&[], &process), true);
}

#[test]
fn with_smaller_map_right_returns_true() {
    is_greater_than_or_equal(
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
fn with_same_size_map_with_greater_keys_returns_true() {
    is_greater_than_or_equal(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("a", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_greater_values_returns_true() {
    is_greater_than_or_equal(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_same_map_returns_true() {
    is_greater_than_or_equal(|left, _| left, true);
}

#[test]
fn with_same_value_map_returns_true() {
    is_greater_than_or_equal(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_greater_values_returns_false() {
    is_greater_than_or_equal(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        4.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_same_size_map_with_greater_keys_returns_false() {
    is_greater_than_or_equal(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("d", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_greater_size_map_returns_false() {
    is_greater_than_or_equal(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("a", DoNotCare).unwrap(),
                        1.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_map_right_returns_true() {
    is_greater_than_or_equal(|_, process| Term::slice_to_map(&[], &process), true);
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
    super::is_greater_than_or_equal(
        |process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        right,
        expected,
    );
}

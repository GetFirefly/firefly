use super::*;

#[test]
fn with_small_integer_right_returns_false() {
    is_less_than(|_, mut process| 0.into_process(&mut process), false)
}

#[test]
fn with_big_integer_right_returns_false() {
    is_less_than(
        |_, mut process| (crate::integer::small::MAX + 1).into_process(&mut process),
        false,
    )
}

#[test]
fn with_float_right_returns_false() {
    is_less_than(|_, mut process| 0.0.into_process(&mut process), false)
}

#[test]
fn with_atom_returns_false() {
    is_less_than(|_, _| Term::str_to_atom("right", DoNotCare).unwrap(), false);
}

#[test]
fn with_local_reference_right_returns_false() {
    is_less_than(|_, mut process| Term::local_reference(&mut process), false);
}

#[test]
fn with_local_pid_right_returns_false() {
    is_less_than(|_, _| Term::local_pid(0, 1).unwrap(), false);
}

#[test]
fn with_external_pid_right_returns_false() {
    is_less_than(
        |_, mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        false,
    );
}

#[test]
fn with_tuple_right_returns_false() {
    is_less_than(
        |_, mut process| Term::slice_to_tuple(&[], &mut process),
        false,
    );
}

#[test]
fn with_smaller_map_right_returns_false() {
    is_less_than(
        |_, mut process| {
            Term::slice_to_map(
                &[(
                    Term::str_to_atom("a", DoNotCare).unwrap(),
                    1.into_process(&mut process),
                )],
                &mut process,
            )
        },
        false,
    );
}

#[test]
fn with_same_size_map_with_lesser_keys_returns_false() {
    is_less_than(
        |_, mut process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("a", DoNotCare).unwrap(),
                        2.into_process(&mut process),
                    ),
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        3.into_process(&mut process),
                    ),
                ],
                &mut process,
            )
        },
        false,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_lesser_values_returns_false() {
    is_less_than(
        |_, mut process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&mut process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        2.into_process(&mut process),
                    ),
                ],
                &mut process,
            )
        },
        false,
    );
}

#[test]
fn with_same_map_returns_false() {
    is_less_than(|left, _| left, false);
}

#[test]
fn with_same_value_map_returns_false() {
    is_less_than(
        |_, mut process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&mut process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&mut process),
                    ),
                ],
                &mut process,
            )
        },
        false,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_greater_values_returns_true() {
    is_less_than(
        |_, mut process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        3.into_process(&mut process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        4.into_process(&mut process),
                    ),
                ],
                &mut process,
            )
        },
        true,
    );
}

#[test]
fn with_same_size_map_with_greater_keys_returns_true() {
    is_less_than(
        |_, mut process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        2.into_process(&mut process),
                    ),
                    (
                        Term::str_to_atom("d", DoNotCare).unwrap(),
                        3.into_process(&mut process),
                    ),
                ],
                &mut process,
            )
        },
        true,
    );
}

#[test]
fn with_greater_size_map_returns_true() {
    is_less_than(
        |_, mut process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("a", DoNotCare).unwrap(),
                        1.into_process(&mut process),
                    ),
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&mut process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&mut process),
                    ),
                ],
                &mut process,
            )
        },
        true,
    );
}

#[test]
fn with_map_right_returns_false() {
    is_less_than(
        |_, mut process| Term::slice_to_map(&[], &mut process),
        false,
    );
}

#[test]
fn with_empty_list_right_returns_true() {
    is_less_than(|_, _| Term::EMPTY_LIST, true);
}

#[test]
fn with_list_right_returns_true() {
    is_less_than(
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
    is_less_than(
        |_, mut process| Term::slice_to_binary(&[], &mut process),
        true,
    );
}

#[test]
fn with_subbinary_right_returns_true() {
    is_less_than(|_, mut process| bitstring!(1 :: 1, &mut process), true);
}

fn is_less_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &mut Process) -> Term,
{
    super::is_less_than(
        |mut process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&mut process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&mut process),
                    ),
                ],
                &mut process,
            )
        },
        right,
        expected,
    );
}

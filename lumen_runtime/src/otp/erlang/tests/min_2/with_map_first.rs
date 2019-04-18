use super::*;

#[test]
fn with_small_integer_second_returns_second() {
    min(|_, mut process| 0.into_process(&mut process), Second)
}

#[test]
fn with_big_integer_second_returns_second() {
    min(
        |_, mut process| (crate::integer::small::MAX + 1).into_process(&mut process),
        Second,
    )
}

#[test]
fn with_float_second_returns_second() {
    min(|_, mut process| 0.0.into_process(&mut process), Second)
}

#[test]
fn with_atom_returns_second() {
    min(
        |_, _| Term::str_to_atom("second", DoNotCare).unwrap(),
        Second,
    );
}

#[test]
fn with_local_reference_second_returns_second() {
    min(|_, mut process| Term::local_reference(&mut process), Second);
}

#[test]
fn with_local_pid_second_returns_second() {
    min(|_, _| Term::local_pid(0, 1).unwrap(), Second);
}

#[test]
fn with_external_pid_second_returns_second() {
    min(
        |_, mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        Second,
    );
}

#[test]
fn with_tuple_second_returns_second() {
    min(
        |_, mut process| Term::slice_to_tuple(&[], &mut process),
        Second,
    );
}

#[test]
fn with_smaller_map_second_returns_second() {
    min(
        |_, mut process| {
            Term::slice_to_map(
                &[(
                    Term::str_to_atom("a", DoNotCare).unwrap(),
                    1.into_process(&mut process),
                )],
                &mut process,
            )
        },
        Second,
    );
}

#[test]
fn with_same_size_map_with_lesser_keys_returns_second() {
    min(
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
        Second,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_lesser_values_returns_second() {
    min(
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
        Second,
    );
}

#[test]
fn with_same_map_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_map_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_greater_values_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_same_size_map_with_greater_keys_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_greater_size_map_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_map_second_returns_second() {
    min(
        |_, mut process| Term::slice_to_map(&[], &mut process),
        Second,
    );
}

#[test]
fn with_empty_list_second_returns_first() {
    min(|_, _| Term::EMPTY_LIST, First);
}

#[test]
fn with_list_second_returns_first() {
    min(
        |_, mut process| {
            Term::cons(
                0.into_process(&mut process),
                1.into_process(&mut process),
                &mut process,
            )
        },
        First,
    );
}

#[test]
fn with_heap_binary_second_returns_first() {
    min(
        |_, mut process| Term::slice_to_binary(&[], &mut process),
        First,
    );
}

#[test]
fn with_subbinary_second_returns_first() {
    min(|_, mut process| bitstring!(1 :: 1, &mut process), First);
}

fn min<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &mut Process) -> Term,
{
    super::min(
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
        second,
        which,
    );
}

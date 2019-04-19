use super::*;

#[test]
fn with_small_integer_second_returns_second() {
    min(|_, process| 0.into_process(&process), Second)
}

#[test]
fn with_big_integer_second_returns_second() {
    min(
        |_, process| (crate::integer::small::MAX + 1).into_process(&process),
        Second,
    )
}

#[test]
fn with_float_second_returns_second() {
    min(|_, process| 0.0.into_process(&process), Second)
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
    min(|_, process| Term::local_reference(&process), Second);
}

#[test]
fn with_local_pid_second_returns_second() {
    min(|_, _| Term::local_pid(0, 1).unwrap(), Second);
}

#[test]
fn with_external_pid_second_returns_second() {
    min(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        Second,
    );
}

#[test]
fn with_smaller_tuple_second_returns_second() {
    min(
        |_, process| Term::slice_to_tuple(&[1.into_process(&process)], &process),
        Second,
    );
}

#[test]
fn with_same_size_tuple_with_lesser_elements_returns_second() {
    min(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 1.into_process(&process)],
                &process,
            )
        },
        Second,
    );
}

#[test]
fn with_same_tuple_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_tuple_returns_first() {
    min(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 2.into_process(&process)],
                &process,
            )
        },
        First,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_first() {
    min(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 3.into_process(&process)],
                &process,
            )
        },
        First,
    );
}

#[test]
fn with_greater_size_tuple_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_map_second_returns_first() {
    min(|_, process| Term::slice_to_map(&[], &process), First);
}

#[test]
fn with_empty_list_second_returns_first() {
    min(|_, _| Term::EMPTY_LIST, First);
}

#[test]
fn with_list_second_returns_first() {
    min(
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        First,
    );
}

#[test]
fn with_heap_binary_second_returns_first() {
    min(|_, process| Term::slice_to_binary(&[], &process), First);
}

#[test]
fn with_subbinary_second_returns_first() {
    min(|_, process| bitstring!(1 :: 1, &process), First);
}

fn min<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::min(
        |process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 2.into_process(&process)],
                &process,
            )
        },
        second,
        which,
    );
}

use super::*;

#[test]
fn with_small_integer_second_returns_first() {
    max(|_, mut process| 0.into_process(&mut process), First)
}

#[test]
fn with_big_integer_second_returns_first() {
    max(
        |_, mut process| (crate::integer::small::MAX + 1).into_process(&mut process),
        First,
    )
}

#[test]
fn with_float_second_returns_first() {
    max(|_, mut process| 0.0.into_process(&mut process), First)
}

#[test]
fn with_atom_returns_first() {
    max(|_, _| Term::str_to_atom("meft", DoNotCare).unwrap(), First);
}

#[test]
fn with_local_reference_second_returns_first() {
    max(|_, mut process| Term::local_reference(&mut process), First);
}

#[test]
fn with_local_pid_second_returns_first() {
    max(|_, _| Term::local_pid(0, 1).unwrap(), First);
}

#[test]
fn with_lesser_external_pid_second_returns_first() {
    max(
        |_, mut process| Term::external_pid(1, 1, 3, &mut process).unwrap(),
        First,
    );
}

#[test]
fn with_same_external_pid_second_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_external_pid_second_returns_first() {
    max(
        |_, mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        First,
    );
}

#[test]
fn with_greater_external_pid_second_returns_second() {
    max(
        |_, mut process| Term::external_pid(1, 3, 3, &mut process).unwrap(),
        Second,
    );
}

#[test]
fn with_tuple_second_returns_second() {
    max(
        |_, mut process| Term::slice_to_tuple(&[], &mut process),
        Second,
    );
}

#[test]
fn with_map_second_returns_second() {
    max(
        |_, mut process| Term::slice_to_map(&[], &mut process),
        Second,
    );
}

#[test]
fn with_empty_list_second_returns_second() {
    max(|_, _| Term::EMPTY_LIST, Second);
}

#[test]
fn with_list_second_returns_second() {
    max(
        |_, mut process| {
            Term::cons(
                0.into_process(&mut process),
                1.into_process(&mut process),
                &mut process,
            )
        },
        Second,
    );
}

#[test]
fn with_heap_binary_second_returns_second() {
    max(
        |_, mut process| Term::slice_to_binary(&[], &mut process),
        Second,
    );
}

#[test]
fn with_subbinary_second_returns_second() {
    max(|_, mut process| bitstring!(1 :: 1, &mut process), Second);
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &mut Process) -> Term,
{
    super::max(
        |mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        second,
        which,
    );
}

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
fn with_external_pid_second_returns_first() {
    max(
        |_, mut process| Term::external_pid(1, 2, 3, &mut process).unwrap(),
        First,
    );
}

#[test]
fn with_tuple_second_returns_first() {
    max(
        |_, mut process| Term::slice_to_tuple(&[], &mut process),
        First,
    );
}

#[test]
fn with_map_second_returns_first() {
    max(
        |_, mut process| Term::slice_to_map(&[], &mut process),
        First,
    );
}

#[test]
fn with_empty_list_second_returns_first() {
    max(|_, _| Term::EMPTY_LIST, First);
}

#[test]
fn with_list_second_returns_first() {
    max(
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
fn with_prefix_heap_binary_second_returns_first() {
    max(
        |_, mut process| Term::slice_to_binary(&[1], &mut process),
        First,
    );
}

#[test]
fn with_same_length_heap_binary_with_lesser_byte_second_returns_first() {
    max(
        |_, mut process| Term::slice_to_binary(&[0], &mut process),
        First,
    );
}

#[test]
fn with_longer_heap_binary_with_lesser_byte_second_returns_first() {
    max(
        |_, mut process| Term::slice_to_binary(&[0, 1, 2], &mut process),
        First,
    );
}

#[test]
fn with_same_heap_binary_second_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_heap_binary_second_returns_first() {
    max(
        |_, mut process| Term::slice_to_binary(&[1, 1], &mut process),
        First,
    )
}

#[test]
fn with_shorter_heap_binary_with_greater_byte_second_returns_second() {
    max(
        |_, mut process| Term::slice_to_binary(&[2], &mut process),
        Second,
    );
}

#[test]
fn with_heap_binary_with_greater_byte_second_returns_second() {
    max(
        |_, mut process| Term::slice_to_binary(&[2, 1], &mut process),
        Second,
    );
}

#[test]
fn with_heap_binary_with_different_greater_byte_second_returns_second() {
    max(
        |_, mut process| Term::slice_to_binary(&[1, 2], &mut process),
        Second,
    );
}

#[test]
fn with_prefix_subbinary_second_returns_first() {
    max(
        |_, mut process| {
            let original = Term::slice_to_binary(&[1], &mut process);
            Term::subbinary(original, 0, 0, 1, 0, &mut process)
        },
        First,
    );
}

#[test]
fn with_same_length_subbinary_with_lesser_byte_second_returns_first() {
    max(
        |_, mut process| {
            let original = Term::slice_to_binary(&[0, 1], &mut process);
            Term::subbinary(original, 0, 0, 2, 0, &mut process)
        },
        First,
    );
}

#[test]
fn with_longer_subbinary_with_lesser_byte_second_returns_first() {
    max(
        |_, mut process| bitstring!(0, 1, 0b10 :: 2, &mut process),
        First,
    );
}

#[test]
fn with_same_subbinary_second_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_subbinary_second_returns_first() {
    max(
        |_, mut process| {
            let original = Term::slice_to_binary(&[1, 1], &mut process);
            Term::subbinary(original, 0, 0, 2, 0, &mut process)
        },
        First,
    )
}

#[test]
fn with_shorter_subbinary_with_greater_byte_second_returns_second() {
    max(
        |_, mut process| {
            let original = Term::slice_to_binary(&[2], &mut process);
            Term::subbinary(original, 0, 0, 1, 0, &mut process)
        },
        Second,
    );
}

#[test]
fn with_subbinary_with_greater_byte_second_returns_second() {
    max(
        |_, mut process| {
            let original = Term::slice_to_binary(&[2, 1], &mut process);
            Term::subbinary(original, 0, 0, 2, 0, &mut process)
        },
        Second,
    );
}

#[test]
fn with_subbinary_with_different_greater_byte_second_returns_second() {
    max(
        |_, mut process| {
            let original = Term::slice_to_binary(&[1, 2], &mut process);
            Term::subbinary(original, 0, 0, 2, 0, &mut process)
        },
        Second,
    );
}

#[test]
fn with_subbinary_with_value_with_shorter_length_returns_second() {
    max(|_, mut process| bitstring!(1, 1 :: 1, &mut process), Second)
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &mut Process) -> Term,
{
    super::max(
        |mut process| Term::slice_to_binary(&[1, 1], &mut process),
        second,
        which,
    );
}

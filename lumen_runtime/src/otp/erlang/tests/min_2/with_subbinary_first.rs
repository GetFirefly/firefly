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
    min(|_, _| Term::str_to_atom("meft", DoNotCare).unwrap(), Second);
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
fn with_tuple_second_returns_second() {
    min(|_, process| Term::slice_to_tuple(&[], &process), Second);
}

#[test]
fn with_map_second_returns_second() {
    min(|_, process| Term::slice_to_map(&[], &process), Second);
}

#[test]
fn with_empty_list_second_returns_second() {
    min(|_, _| Term::EMPTY_LIST, Second);
}

#[test]
fn with_list_second_returns_second() {
    min(
        |_, process| Term::cons(0.into_process(&process), 1.into_process(&process), &process),
        Second,
    );
}

#[test]
fn with_prefix_heap_binary_second_returns_second() {
    min(|_, process| Term::slice_to_binary(&[1], &process), Second);
}

#[test]
fn with_same_length_heap_binary_with_lesser_byte_second_returns_second() {
    min(|_, process| Term::slice_to_binary(&[0], &process), Second);
}

#[test]
fn with_longer_heap_binary_with_lesser_byte_second_returns_second() {
    min(
        |_, process| Term::slice_to_binary(&[0, 1, 2], &process),
        Second,
    );
}

#[test]
fn with_same_value_heap_binary_second_returns_first() {
    super::min(
        |process| {
            let original = Term::slice_to_binary(&[1], &process);
            Term::subbinary(original, 0, 0, 1, 0, &process)
        },
        |_, process| Term::slice_to_binary(&[1], &process),
        First,
    )
}

#[test]
fn with_shorter_heap_binary_with_greater_byte_second_returns_first() {
    min(|_, process| Term::slice_to_binary(&[2], &process), First);
}

#[test]
fn with_heap_binary_with_greater_byte_second_returns_first() {
    min(|_, process| Term::slice_to_binary(&[2, 1], &process), First);
}

#[test]
fn with_heap_binary_with_greater_byte_than_bits_second_returns_first() {
    min(
        |_, process| Term::slice_to_binary(&[1, 0b1000_0000], &process),
        First,
    );
}

#[test]
fn with_prefix_subbinary_second_returns_second() {
    min(
        |_, process| {
            let original = Term::slice_to_binary(&[1], &process);
            Term::subbinary(original, 0, 0, 1, 0, &process)
        },
        Second,
    );
}

#[test]
fn with_same_length_subbinary_with_lesser_byte_second_returns_second() {
    min(
        |_, process| {
            let original = Term::slice_to_binary(&[0, 1], &process);
            Term::subbinary(original, 0, 0, 2, 0, &process)
        },
        Second,
    );
}

#[test]
fn with_longer_subbinary_with_lesser_byte_second_returns_second() {
    min(|_, process| bitstring!(0, 1, 0b10 :: 2, &process), Second);
}

#[test]
fn with_same_subbinary_second_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_subbinary_second_returns_first() {
    min(|_, process| bitstring!(1, 1 :: 2, &process), First);
}

#[test]
fn with_shorter_subbinary_with_greater_byte_second_returns_first() {
    min(
        |_, process| {
            let original = Term::slice_to_binary(&[2], &process);
            Term::subbinary(original, 0, 0, 1, 0, &process)
        },
        First,
    );
}

#[test]
fn with_subbinary_with_greater_byte_second_returns_first() {
    min(
        |_, process| {
            let original = Term::slice_to_binary(&[2, 1], &process);
            Term::subbinary(original, 0, 0, 2, 0, &process)
        },
        First,
    );
}

#[test]
fn with_subbinary_with_different_greater_byte_second_returns_first() {
    min(
        |_, process| {
            let original = Term::slice_to_binary(&[1, 2], &process);
            Term::subbinary(original, 0, 0, 2, 0, &process)
        },
        First,
    );
}

#[test]
fn with_subbinary_with_value_with_shorter_length_returns_first() {
    min(|_, process| bitstring!(1, 1 :: 1, &process), First)
}

fn min<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::min(|process| bitstring!(1, 1 :: 2, &process), second, which);
}

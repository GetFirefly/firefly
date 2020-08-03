use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_pid_tuple_map_or_list_second_returns_first() {
    run!(
        |arc_process| {
            (
                    strategy::term::binary::heap(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be number, atom, reference, function, port, pid, tuple, map, or list",
                        |second| {
                            second.is_number()
                                || second.is_atom()
                                || second.is_reference()
                                || second.is_boxed_function()
                                || second.is_port()
                                || second.is_pid()
                                || second.is_boxed_tuple()
                                || second.is_list()
                        }),
                )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), first);

            Ok(())
        },
    );
}

#[test]
fn with_prefix_heap_binary_second_returns_first() {
    max(|_, process| process.binary_from_bytes(&[1]).unwrap(), First);
}

#[test]
fn with_same_length_heap_binary_with_lesser_byte_second_returns_first() {
    max(|_, process| process.binary_from_bytes(&[0]).unwrap(), First);
}

#[test]
fn with_longer_heap_binary_with_lesser_byte_second_returns_first() {
    max(
        |_, process| process.binary_from_bytes(&[0, 1, 2]).unwrap(),
        First,
    );
}

#[test]
fn with_same_value_heap_binary_second_returns_first() {
    super::max(
        |process| {
            let original = process.binary_from_bytes(&[1]).unwrap();
            process
                .subbinary_from_original(original, 0, 0, 1, 0)
                .unwrap()
        },
        |_, process| process.binary_from_bytes(&[1]).unwrap(),
        First,
    )
}

#[test]
fn with_shorter_heap_binary_with_greater_byte_second_returns_second() {
    max(
        |_, process| process.binary_from_bytes(&[2]).unwrap(),
        Second,
    );
}

#[test]
fn with_heap_binary_with_greater_byte_second_returns_second() {
    max(
        |_, process| process.binary_from_bytes(&[2, 1]).unwrap(),
        Second,
    );
}

#[test]
fn with_heap_binary_with_greater_byte_than_bits_second_returns_second() {
    max(
        |_, process| process.binary_from_bytes(&[1, 0b1000_0000]).unwrap(),
        Second,
    );
}

#[test]
fn with_prefix_subbinary_second_returns_first() {
    max(
        |_, process| {
            let original = process.binary_from_bytes(&[1]).unwrap();
            process
                .subbinary_from_original(original, 0, 0, 1, 0)
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_length_subbinary_with_lesser_byte_second_returns_first() {
    max(
        |_, process| {
            let original = process.binary_from_bytes(&[0, 1]).unwrap();
            process
                .subbinary_from_original(original, 0, 0, 2, 0)
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_longer_subbinary_with_lesser_byte_second_returns_first() {
    max(|_, process| bitstring!(0, 1, 0b10 :: 2, &process), First);
}

#[test]
fn with_same_subbinary_second_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_subbinary_second_returns_first() {
    max(|_, process| bitstring!(1, 1 :: 2, &process), First);
}

#[test]
fn with_shorter_subbinary_with_greater_byte_second_returns_second() {
    max(
        |_, process| {
            let original = process.binary_from_bytes(&[2]).unwrap();
            process
                .subbinary_from_original(original, 0, 0, 1, 0)
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_subbinary_with_greater_byte_second_returns_second() {
    max(
        |_, process| {
            let original = process.binary_from_bytes(&[2, 1]).unwrap();
            process
                .subbinary_from_original(original, 0, 0, 2, 0)
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_subbinary_with_different_greater_byte_second_returns_second() {
    max(
        |_, process| {
            let original = process.binary_from_bytes(&[1, 2]).unwrap();
            process
                .subbinary_from_original(original, 0, 0, 2, 0)
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_subbinary_with_value_with_shorter_length_returns_second() {
    max(|_, process| bitstring!(1, 1 :: 1, &process), Second)
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::max(|process| bitstring!(1, 1 :: 2, &process), second, which);
}

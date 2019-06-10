use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_pid_tuple_map_or_list_second_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::heap(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be number, atom, reference, function, port, pid, tuple, map, or list",
                        |second| {
                            second.is_number()
                                || second.is_atom()
                                || second.is_reference()
                                || second.is_function()
                                || second.is_port()
                                || second.is_pid()
                                || second.is_tuple()
                                || second.is_list()
                        }),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::max_2(first, second), first);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_prefix_heap_binary_second_returns_first() {
    max(|_, process| Term::slice_to_binary(&[1], &process), First);
}

#[test]
fn with_same_length_heap_binary_with_lesser_byte_second_returns_first() {
    max(|_, process| Term::slice_to_binary(&[0], &process), First);
}

#[test]
fn with_longer_heap_binary_with_lesser_byte_second_returns_first() {
    max(
        |_, process| Term::slice_to_binary(&[0, 1, 2], &process),
        First,
    );
}

#[test]
fn with_same_value_heap_binary_second_returns_first() {
    super::max(
        |process| {
            let original = Term::slice_to_binary(&[1], &process);
            Term::subbinary(original, 0, 0, 1, 0, &process)
        },
        |_, process| Term::slice_to_binary(&[1], &process),
        First,
    )
}

#[test]
fn with_shorter_heap_binary_with_greater_byte_second_returns_second() {
    max(|_, process| Term::slice_to_binary(&[2], &process), Second);
}

#[test]
fn with_heap_binary_with_greater_byte_second_returns_second() {
    max(
        |_, process| Term::slice_to_binary(&[2, 1], &process),
        Second,
    );
}

#[test]
fn with_heap_binary_with_greater_byte_than_bits_second_returns_second() {
    max(
        |_, process| Term::slice_to_binary(&[1, 0b1000_0000], &process),
        Second,
    );
}

#[test]
fn with_prefix_subbinary_second_returns_first() {
    max(
        |_, process| {
            let original = Term::slice_to_binary(&[1], &process);
            Term::subbinary(original, 0, 0, 1, 0, &process)
        },
        First,
    );
}

#[test]
fn with_same_length_subbinary_with_lesser_byte_second_returns_first() {
    max(
        |_, process| {
            let original = Term::slice_to_binary(&[0, 1], &process);
            Term::subbinary(original, 0, 0, 2, 0, &process)
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
            let original = Term::slice_to_binary(&[2], &process);
            Term::subbinary(original, 0, 0, 1, 0, &process)
        },
        Second,
    );
}

#[test]
fn with_subbinary_with_greater_byte_second_returns_second() {
    max(
        |_, process| {
            let original = Term::slice_to_binary(&[2, 1], &process);
            Term::subbinary(original, 0, 0, 2, 0, &process)
        },
        Second,
    );
}

#[test]
fn with_subbinary_with_different_greater_byte_second_returns_second() {
    max(
        |_, process| {
            let original = Term::slice_to_binary(&[1, 2], &process);
            Term::subbinary(original, 0, 0, 2, 0, &process)
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

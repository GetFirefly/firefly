use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::binary_to_list_3::result;
use crate::test::strategy;
use crate::test::strategy::NON_EMPTY_RANGE_INCLUSIVE;

#[test]
fn without_binary_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_binary(arc_process.clone()),
            )
        },
        |(arc_process, binary)| {
            let start = arc_process.integer(1).unwrap();
            let stop = arc_process.integer(1).unwrap();

            prop_assert_badarg!(
                result(&arc_process, binary, start, stop),
                format!("binary ({}) must be a binary", binary)
            );

            Ok(())
        },
    );
}

#[test]
fn with_binary_without_integer_start_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_binary(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, binary, start)| {
            let stop = arc_process.integer(1).unwrap();

            prop_assert_badarg!(
                        result(&arc_process, binary, start, stop),
                        format!("start ({}) must be a one-based integer index between 1 and the byte size of the binary", start)
                    );

            Ok(())
        },
    );
}

#[test]
fn with_binary_with_positive_integer_start_without_integer_stop_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_binary(arc_process.clone()),
                strategy::term::integer::positive(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, binary, start, stop)| {
            prop_assert_badarg!(
                        result(&arc_process, binary, start, stop),
                        format!("stop ({}) must be a one-based integer index between 1 and the byte size of the binary", stop)
                    );

            Ok(())
        },
    );
}

#[test]
fn with_binary_with_start_less_than_or_equal_to_stop_returns_list_of_bytes() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::byte_vec::with_size_range(NON_EMPTY_RANGE_INCLUSIVE.into()),
            )
                .prop_flat_map(|(arc_process, byte_vec)| {
                    let max_start = byte_vec.len();

                    (
                        Just(arc_process.clone()),
                        Just(byte_vec.clone()),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                        (1..=max_start),
                    )
                })
                .prop_flat_map(|(arc_process, byte_vec, binary, start)| {
                    let max_stop = byte_vec.len();

                    (
                        Just(arc_process.clone()),
                        Just(byte_vec),
                        Just(binary),
                        Just(start),
                        start..=max_stop,
                    )
                })
        },
        |(arc_process, byte_vec, binary, start, stop)| {
            let (list, start_term, stop_term) = {
                // not using an iterator because that would too closely match the code under
                // test
                let list = match (start, stop) {
                    (1, 1) => arc_process
                        .cons(arc_process.integer(byte_vec[0]).unwrap(), Term::NIL)
                        .unwrap(),
                    (1, 2) => arc_process
                        .cons(
                            arc_process.integer(byte_vec[0]).unwrap(),
                            arc_process
                                .cons(arc_process.integer(byte_vec[1]).unwrap(), Term::NIL)
                                .unwrap(),
                        )
                        .unwrap(),
                    (1, 3) => arc_process
                        .cons(
                            arc_process.integer(byte_vec[0]).unwrap(),
                            arc_process
                                .cons(
                                    arc_process.integer(byte_vec[1]).unwrap(),
                                    arc_process
                                        .cons(arc_process.integer(byte_vec[2]).unwrap(), Term::NIL)
                                        .unwrap(),
                                )
                                .unwrap(),
                        )
                        .unwrap(),
                    (2, 2) => arc_process
                        .cons(arc_process.integer(byte_vec[1]).unwrap(), Term::NIL)
                        .unwrap(),
                    (2, 3) => arc_process
                        .cons(
                            arc_process.integer(byte_vec[1]).unwrap(),
                            arc_process
                                .cons(arc_process.integer(byte_vec[2]).unwrap(), Term::NIL)
                                .unwrap(),
                        )
                        .unwrap(),
                    (3, 3) => arc_process
                        .cons(arc_process.integer(byte_vec[2]).unwrap(), Term::NIL)
                        .unwrap(),
                    _ => unimplemented!("(start, stop) = ({:?}, {:?})", start, stop),
                };
                let start_term = arc_process.integer(start).unwrap();
                let stop_term = arc_process.integer(stop).unwrap();

                (list, start_term, stop_term)
            };

            prop_assert_eq!(
                result(&arc_process, binary, start_term, stop_term),
                Ok(list)
            );

            Ok(())
        },
    );
}

#[test]
fn with_binary_with_start_greater_than_stop_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::byte_vec::with_size_range((2..=4).into()),
            )
                .prop_flat_map(|(arc_process, byte_vec)| {
                    // -1 so that start can be greater
                    let max_stop = byte_vec.len() - 1;

                    (
                        Just(arc_process.clone()),
                        Just(byte_vec.len()),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                        (1..=max_stop),
                    )
                })
                .prop_flat_map(|(arc_process, max_start, binary, stop)| {
                    (
                        Just(arc_process.clone()),
                        Just(binary),
                        (stop + 1)..=max_start,
                        Just(stop),
                    )
                })
        },
        |(arc_process, binary, start, stop)| {
            let (start_term, stop_term) = {
                let mut heap = arc_process.acquire_heap();

                let start_term = heap.integer(start).unwrap();
                let stop_term = heap.integer(stop).unwrap();

                (start_term, stop_term)
            };

            prop_assert_badarg!(
                result(&arc_process, binary, start_term, stop_term),
                format!(
                    "start ({}) must be less than or equal to stop ({})",
                    start, stop
                )
            );

            Ok(())
        },
    );
}

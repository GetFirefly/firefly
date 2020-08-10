use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::process::alloc::TermAlloc;

use crate::erlang::binary_to_list_3::result;
use crate::test::strategy;

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

// `with_binary_with_start_less_than_or_equal_to_stop_returns_list_of_bytes` in integration tests

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

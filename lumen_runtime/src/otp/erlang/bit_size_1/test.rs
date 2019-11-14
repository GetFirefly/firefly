use std::convert::TryInto;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::bit_size_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(native(&arc_process, binary), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_heap_binary_is_eight_times_byte_count() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::heap(arc_process.clone()),
                |binary| {
                    let result = native(&arc_process, binary);

                    prop_assert!(result.is_ok());

                    let bit_size_term = result.unwrap();
                    let bit_size_small_integer: SmallInteger = bit_size_term.try_into().unwrap();
                    let bit_size: usize = bit_size_small_integer.try_into().unwrap();

                    prop_assert_eq!(bit_size % 8, 0);

                    let heap_binary: Boxed<HeapBin> = binary.try_into().unwrap();

                    prop_assert_eq!(heap_binary.total_byte_len() * 8, bit_size);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_subbinary_is_eight_times_byte_count_plus_bit_count() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::sub(arc_process.clone()),
                |binary| {
                    let result = native(&arc_process, binary);

                    prop_assert!(result.is_ok());

                    let bit_size_term = result.unwrap();
                    let bit_size_small_integer: SmallInteger = bit_size_term.try_into().unwrap();
                    let bit_size: usize = bit_size_small_integer.try_into().unwrap();

                    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();

                    prop_assert_eq!(
                        subbinary.full_byte_len() * 8 + subbinary.partial_byte_bit_len() as usize,
                        bit_size
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

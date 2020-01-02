use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::byte_size_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |bitstring| {
                    prop_assert_badarg!(
                        native(&arc_process, bitstring),
                        format!("bitstring ({}) is not a bitstring", bitstring)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_heap_binary_is_byte_count() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::byte_vec().prop_map(|byte_vec| {
                    (
                        byte_vec.len(),
                        arc_process.binary_from_bytes(&byte_vec).unwrap(),
                    )
                }),
                |(byte_count, bitstring)| {
                    prop_assert_eq!(
                        native(&arc_process, bitstring),
                        Ok(arc_process.integer(byte_count).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_subbinary_without_bit_count_is_byte_count() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::byte_vec().prop_flat_map(|byte_vec| {
                    (
                        Just(byte_vec.len()),
                        strategy::term::binary::sub::containing_bytes(
                            byte_vec,
                            arc_process.clone(),
                        ),
                    )
                }),
                |(byte_count, bitstring)| {
                    prop_assert_eq!(
                        native(&arc_process, bitstring),
                        Ok(arc_process.integer(byte_count).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_subbinary_with_bit_count_is_byte_count_plus_one() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::binary::sub::byte_count()).prop_flat_map(|byte_count| {
                    (
                        Just(byte_count),
                        strategy::term::binary::sub::with_size_range(
                            strategy::term::binary::sub::byte_offset(),
                            strategy::term::binary::sub::bit_offset(),
                            (byte_count..=byte_count).boxed(),
                            (1_u8..=7_u8).boxed(),
                            arc_process.clone(),
                        ),
                    )
                }),
                |(byte_count, bitstring)| {
                    prop_assert_eq!(
                        native(&arc_process, bitstring),
                        Ok(arc_process.integer(byte_count + 1).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

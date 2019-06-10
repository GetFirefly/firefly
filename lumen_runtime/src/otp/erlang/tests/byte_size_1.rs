use super::*;

use proptest::strategy::Strategy;

use crate::process::IntoProcess;

#[test]
fn without_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |bitstring| {
                    prop_assert_eq!(erlang::byte_size_1(bitstring, &arc_process), Err(badarg!()));

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
                        Term::slice_to_binary(&byte_vec, &arc_process),
                    )
                }),
                |(byte_count, bitstring)| {
                    prop_assert_eq!(
                        erlang::byte_size_1(bitstring, &arc_process),
                        Ok(byte_count.into_process(&arc_process))
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
                        erlang::byte_size_1(bitstring, &arc_process),
                        Ok(byte_count.into_process(&arc_process))
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
                        erlang::byte_size_1(bitstring, &arc_process),
                        Ok((byte_count + 1).into_process(&arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

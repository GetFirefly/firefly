use super::*;

#[test]
fn without_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(erlang::bit_size_1(binary, &arc_process), Err(badarg!()));

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
                    let result = erlang::bit_size_1(binary, &arc_process);

                    prop_assert!(result.is_ok());

                    let bit_size_term = result.unwrap();
                    let bit_size = unsafe { bit_size_term.small_integer_to_isize() } as usize;

                    prop_assert_eq!(bit_size % 8, 0);

                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    prop_assert_eq!(heap_binary.byte_len() * 8, bit_size);

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
                    let result = erlang::bit_size_1(binary, &arc_process);

                    prop_assert!(result.is_ok());

                    let bit_size_term = result.unwrap();
                    let bit_size = unsafe { bit_size_term.small_integer_to_isize() } as usize;

                    let subbinary: &sub::Binary = binary.unbox_reference();

                    prop_assert_eq!(
                        subbinary.byte_count * 8 + subbinary.bit_count as usize,
                        bit_size
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

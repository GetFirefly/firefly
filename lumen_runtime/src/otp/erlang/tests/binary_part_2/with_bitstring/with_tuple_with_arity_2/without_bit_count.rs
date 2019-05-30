use super::*;

#[test]
fn with_positive_start_and_positive_length_returns_subbinary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_binary::with_byte_len_range(
                    (3..=6).into(),
                    arc_process.clone(),
                )
                .prop_flat_map(|binary| {
                    let byte_len = binary.byte_len();

                    // `start` must be 2 less than `byte_len` so that `length` can be at least 1
                    (Just(binary), (1..=(byte_len - 2)))
                })
                .prop_flat_map(|(binary, start)| {
                    (Just(binary), Just(start), 1..=(binary.byte_len() - start))
                })
                .prop_map(|(binary, start, length)| {
                    (
                        binary,
                        start.into_process(&arc_process),
                        length.into_process(&arc_process),
                    )
                }),
                |(binary, start, length)| {
                    let start_length = Term::slice_to_tuple(&[start, length], &arc_process);

                    let result = erlang::binary_part_2(binary, start_length, &arc_process);

                    prop_assert!(result.is_ok());

                    let returned_boxed = result.unwrap();

                    prop_assert_eq!(returned_boxed.tag(), Boxed);

                    let returned_unboxed: &Term = returned_boxed.unbox_reference();

                    prop_assert_eq!(returned_unboxed.tag(), Subbinary);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_size_start_and_negative_size_length_returns_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_binary::with_byte_len_range(1..=4, arc_process.clone())
                    .prop_map(|binary| {
                        let byte_len = binary.byte_len();

                        (
                            binary,
                            byte_len.into_process(&arc_process),
                            (-(byte_len as isize)).into_process(&arc_process),
                        )
                    }),
                |(binary, start, length)| {
                    let start_length = Term::slice_to_tuple(&[start, length], &arc_process);

                    prop_assert_eq!(
                        erlang::binary_part_2(binary, start_length, &arc_process),
                        Ok(binary)
                    );

                    let returned_binary =
                        erlang::binary_part_2(binary, start_length, &arc_process).unwrap();

                    prop_assert_eq!(returned_binary.tagged, binary.tagged);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_zero_start_and_size_length_returns_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_binary(arc_process.clone()).prop_map(|binary| {
                    (
                        binary,
                        0.into_process(&arc_process),
                        binary.byte_len().into_process(&arc_process),
                    )
                }),
                |(binary, start, length)| {
                    let start_length = Term::slice_to_tuple(&[start, length], &arc_process);

                    prop_assert_eq!(
                        erlang::binary_part_2(binary, start_length, &arc_process),
                        Ok(binary)
                    );

                    let returned_binary =
                        erlang::binary_part_2(binary, start_length, &arc_process).unwrap();

                    prop_assert_eq!(returned_binary.tagged, binary.tagged);

                    Ok(())
                },
            )
            .unwrap();
    });
}

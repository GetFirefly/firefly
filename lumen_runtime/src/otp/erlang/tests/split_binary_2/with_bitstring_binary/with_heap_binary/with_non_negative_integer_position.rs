use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_less_than_byte_len_returns_binary_prefix_and_suffix_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(2_usize..=4_usize).prop_flat_map(|byte_len| {
                    (
                        strategy::byte_vec::with_size_range((byte_len..=byte_len).into()),
                        1..byte_len,
                    )
                }),
                |(byte_vec, index)| {
                    let binary = Term::slice_to_binary(&byte_vec, &arc_process);
                    let position = index.into_process(&arc_process);

                    let prefix_bytes = &byte_vec[0..index];
                    let prefix = Term::slice_to_binary(prefix_bytes, &arc_process);

                    let suffix_bytes = &byte_vec[index..];
                    let suffix = Term::slice_to_binary(suffix_bytes, &arc_process);

                    prop_assert_eq!(
                        erlang::split_binary_2(binary, position, &arc_process),
                        Ok(Term::slice_to_tuple(&[prefix, suffix], &arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_byte_len_returns_subbinary_and_empty_suffix() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::byte_vec(), |byte_vec| {
                let binary = Term::slice_to_binary(&byte_vec, &arc_process);
                let position = byte_vec.len().into_process(&arc_process);

                prop_assert_eq!(
                    erlang::split_binary_2(binary, position, &arc_process),
                    Ok(Term::slice_to_tuple(
                        &[binary, Term::slice_to_binary(&[], &arc_process)],
                        &arc_process
                    ))
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_greater_than_byte_len_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::byte_vec().prop_flat_map(|byte_vec| {
                    let min = byte_vec.len() + 1;
                    let max = std::isize::MAX as usize;

                    (Just(byte_vec), min..=max)
                }),
                |(byte_vec, index)| {
                    let binary = Term::slice_to_binary(&byte_vec, &arc_process);
                    let position = index.into_process(&arc_process);

                    prop_assert_eq!(
                        erlang::split_binary_2(binary, position, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

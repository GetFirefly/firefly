use super::*;

#[test]
fn with_binary_encoding_atom_that_does_not_exist_errors_badarg() {
    // :erlang.term_to_binary(:non_existent_0)
    let byte_vec = vec![
        131, 100, 0, 14, 110, 111, 110, 95, 101, 120, 105, 115, 116, 101, 110, 116, 95, 48,
    ];

    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_term_2(binary, options(&arc_process), &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_encoding_list_containing_atom_that_does_not_exist_errors_badarg() {
    // :erlang.term_to_binary([:non_existent_1])
    let byte_vec = vec![
        131, 108, 0, 0, 0, 1, 100, 0, 14, 110, 111, 110, 95, 101, 120, 105, 115, 116, 101, 110,
        116, 95, 49, 106,
    ];

    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_term_2(binary, options(&arc_process), &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_encoding_small_tuple_containing_atom_that_does_not_exist_errors_badarg() {
    // :erlang.term_to_binary({:non_existent_2})
    let byte_vec = vec![
        131, 104, 1, 100, 0, 14, 110, 111, 110, 95, 101, 120, 105, 115, 116, 101, 110, 116, 95, 50,
    ];

    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_term_2(binary, options(&arc_process), &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_encoding_small_atom_utf8_that_does_not_exist_errors_badarg() {
    // :erlang.term_to_binary(:"non_existent_3_😈")
    let byte_vec = vec![
        131, 119, 19, 110, 111, 110, 95, 101, 120, 105, 115, 116, 101, 110, 116, 95, 51, 95, 240,
        159, 152, 136,
    ];

    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_term_2(binary, options(&arc_process), &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn options(process: &Process) -> Term {
    Term::cons(
        Term::str_to_atom("safe", DoNotCare).unwrap(),
        Term::EMPTY_LIST,
        &process,
    )
}

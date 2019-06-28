use super::*;

mod with_safe;

#[test]
fn with_used_with_binary_returns_how_many_bytes_were_consumed_along_with_term() {
    // <<131,100,0,5,"hello","world">>
    let byte_vec = vec![
        131, 100, 0, 5, 104, 101, 108, 108, 111, 119, 111, 114, 108, 100,
    ];

    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                |binary| {
                    let options = options(&arc_process);
                    let term = Term::str_to_atom("hello", DoNotCare).unwrap();

                    prop_assert_eq!(
                        erlang::binary_to_term_2(binary, options, &arc_process),
                        Ok(Term::slice_to_tuple(
                            &[term, 9.into_process(&arc_process)],
                            &arc_process
                        ))
                    );

                    // Using only `used` portion of binary returns the same result
                    let tuple = erlang::binary_to_term_2(binary, options, &arc_process).unwrap();
                    let used_term =
                        erlang::element_2(2.into_process(&arc_process), tuple, &arc_process)
                            .unwrap();
                    let split_binary_tuple =
                        erlang::split_binary_2(binary, used_term, &arc_process).unwrap();
                    let prefix = erlang::element_2(
                        1.into_process(&arc_process),
                        split_binary_tuple,
                        &arc_process,
                    )
                    .unwrap();

                    prop_assert_eq!(
                        erlang::binary_to_term_2(prefix, options, &arc_process),
                        Ok(tuple)
                    );

                    // Without used returns only term

                    prop_assert_eq!(
                        erlang::binary_to_term_2(binary, Term::EMPTY_LIST, &arc_process),
                        Ok(term)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });

    fn options(process: &Process) -> Term {
        Term::cons(
            Term::str_to_atom("used", DoNotCare).unwrap(),
            Term::EMPTY_LIST,
            process,
        )
    }
}

use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_binary(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_list_1(binary, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_returns_list_of_bytes() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::byte_vec().prop_flat_map(|byte_vec| {
                    (
                        Just(byte_vec.clone()),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                    )
                }),
                |(byte_vec, binary)| {
                    // not using an iterator because that would too closely match the code under
                    // test
                    let list = match byte_vec.len() {
                        0 => Term::EMPTY_LIST,
                        1 => Term::cons(
                            byte_vec[0].into_process(&arc_process),
                            Term::EMPTY_LIST,
                            &arc_process,
                        ),
                        2 => Term::cons(
                            byte_vec[0].into_process(&arc_process),
                            Term::cons(
                                byte_vec[1].into_process(&arc_process),
                                Term::EMPTY_LIST,
                                &arc_process,
                            ),
                            &arc_process,
                        ),
                        3 => Term::cons(
                            byte_vec[0].into_process(&arc_process),
                            Term::cons(
                                byte_vec[1].into_process(&arc_process),
                                Term::cons(
                                    byte_vec[2].into_process(&arc_process),
                                    Term::EMPTY_LIST,
                                    &arc_process,
                                ),
                                &arc_process,
                            ),
                            &arc_process,
                        ),
                        len => unimplemented!("len = {:?}", len),
                    };

                    prop_assert_eq!(erlang::binary_to_list_1(binary, &arc_process), Ok(list));

                    Ok(())
                },
            )
            .unwrap();
    });
}

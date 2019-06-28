use super::*;

use crate::process::IntoProcess;

use crate::otp::erlang::tests::strategy::NON_EMPTY_RANGE_INCLUSIVE;
use proptest::strategy::Strategy;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_binary(arc_process.clone()),
                |binary| {
                    let start = 1.into_process(&arc_process);
                    let stop = 1.into_process(&arc_process);

                    prop_assert_eq!(
                        erlang::binary_to_list_3(binary, start, stop, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_without_integer_start_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_binary(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(binary, start)| {
                    let stop = 1.into_process(&arc_process);

                    prop_assert_eq!(
                        erlang::binary_to_list_3(binary, start, stop, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_integer_start_without_integer_stop_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_binary(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(binary, start, stop)| {
                    prop_assert_eq!(
                        erlang::binary_to_list_3(binary, start, stop, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_start_less_than_or_equal_to_stop_returns_list_of_bytes() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::byte_vec::with_size_range(NON_EMPTY_RANGE_INCLUSIVE.into())
                    .prop_flat_map(|byte_vec| {
                        let max_start = byte_vec.len();

                        (
                            Just(byte_vec.clone()),
                            strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                            (1..=max_start),
                        )
                    })
                    .prop_flat_map(|(byte_vec, binary, start)| {
                        let max_stop = byte_vec.len();

                        (Just(byte_vec), Just(binary), Just(start), start..=max_stop)
                    }),
                |(byte_vec, binary, start, stop)| {
                    // not using an iterator because that would too closely match the code under
                    // test
                    let list = match (start, stop) {
                        (1, 1) => Term::cons(
                            byte_vec[0].into_process(&arc_process),
                            Term::EMPTY_LIST,
                            &arc_process,
                        ),
                        (1, 2) => Term::cons(
                            byte_vec[0].into_process(&arc_process),
                            Term::cons(
                                byte_vec[1].into_process(&arc_process),
                                Term::EMPTY_LIST,
                                &arc_process,
                            ),
                            &arc_process,
                        ),
                        (1, 3) => Term::cons(
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
                        (2, 2) => Term::cons(
                            byte_vec[1].into_process(&arc_process),
                            Term::EMPTY_LIST,
                            &arc_process,
                        ),
                        (2, 3) => Term::cons(
                            byte_vec[1].into_process(&arc_process),
                            Term::cons(
                                byte_vec[2].into_process(&arc_process),
                                Term::EMPTY_LIST,
                                &arc_process,
                            ),
                            &arc_process,
                        ),
                        (3, 3) => Term::cons(
                            byte_vec[2].into_process(&arc_process),
                            Term::EMPTY_LIST,
                            &arc_process,
                        ),
                        _ => unimplemented!("(start, stop) = ({:?}, {:?})", start, stop),
                    };

                    prop_assert_eq!(
                        erlang::binary_to_list_3(
                            binary,
                            start.into_process(&arc_process),
                            stop.into_process(&arc_process),
                            &arc_process
                        ),
                        Ok(list)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_start_greater_than_stop_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::byte_vec::with_size_range((2..=4).into())
                    .prop_flat_map(|byte_vec| {
                        // -1 so that start can be greater
                        let max_stop = byte_vec.len() - 1;

                        (
                            Just(byte_vec.len()),
                            strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                            (1..=max_stop),
                        )
                    })
                    .prop_flat_map(|(max_start, binary, stop)| {
                        (Just(binary), (stop + 1)..=max_start, Just(stop))
                    }),
                |(binary, start, stop)| {
                    prop_assert_eq!(
                        erlang::binary_to_list_3(
                            binary,
                            start.into_process(&arc_process),
                            stop.into_process(&arc_process),
                            &arc_process
                        ),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

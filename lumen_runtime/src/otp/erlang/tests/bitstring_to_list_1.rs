use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |bitstring| {
                    prop_assert_eq!(
                        erlang::bitstring_to_list_1(bitstring, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_heap_binary_returns_list_of_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::byte_vec(), |byte_vec| {
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

                let bitstring = Term::slice_to_binary(&byte_vec, &arc_process);

                prop_assert_eq!(
                    erlang::bitstring_to_list_1(bitstring, &arc_process),
                    Ok(list)
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_subbinary_without_bit_count_returns_list_of_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::byte_vec().prop_flat_map(|byte_vec| {
                    (
                        Just(byte_vec.clone()),
                        strategy::term::binary::sub::containing_bytes(
                            byte_vec,
                            arc_process.clone(),
                        ),
                    )
                }),
                |(byte_vec, bitstring)| {
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

                    prop_assert_eq!(
                        erlang::bitstring_to_list_1(bitstring, &arc_process),
                        Ok(list)
                    );

                    Ok(())
                },
            )
            .unwrap();
    })
}

#[test]
fn with_subbinary_with_bit_count_returns_list_of_integer_with_bitstring_for_bit_count() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::sub::is_not_binary(arc_process.clone()),
                |bitstring| {
                    let subbinary: &sub::Binary = bitstring.unbox_reference();

                    let byte_vec: Vec<u8> = subbinary.byte_iter().collect();

                    let mut bit_count_byte: u8 = 0;

                    for (i, bit) in subbinary.bit_count_iter().enumerate() {
                        bit_count_byte = bit_count_byte | (bit << (7 - i));
                    }

                    let bits_original_byte_vec = vec![bit_count_byte];
                    let bits_original =
                        Term::slice_to_binary(&bits_original_byte_vec, &arc_process);
                    let bits_subbinary =
                        Term::subbinary(bits_original, 0, 0, 0, subbinary.bit_count, &arc_process);

                    // not using an iterator because that would too closely match the code under
                    // test
                    let list = match byte_vec.len() {
                        0 => Term::cons(bits_subbinary, Term::EMPTY_LIST, &arc_process),
                        1 => Term::cons(
                            byte_vec[0].into_process(&arc_process),
                            Term::cons(bits_subbinary, Term::EMPTY_LIST, &arc_process),
                            &arc_process,
                        ),
                        2 => Term::cons(
                            byte_vec[0].into_process(&arc_process),
                            Term::cons(
                                byte_vec[1].into_process(&arc_process),
                                Term::cons(bits_subbinary, Term::EMPTY_LIST, &arc_process),
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
                                    Term::cons(bits_subbinary, Term::EMPTY_LIST, &arc_process),
                                    &arc_process,
                                ),
                                &arc_process,
                            ),
                            &arc_process,
                        ),
                        len => unimplemented!("len = {:?}", len),
                    };

                    prop_assert_eq!(
                        erlang::bitstring_to_list_1(bitstring, &arc_process),
                        Ok(list)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

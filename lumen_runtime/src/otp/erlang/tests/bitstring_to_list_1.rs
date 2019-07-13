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
                        Err(badarg!().into())
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
                let list = {
                    // not using an iterator because that would too closely match the code under
                    // test
                    match byte_vec.len() {
                        0 => Term::NIL,
                        1 => arc_process
                            .cons(arc_process.integer(byte_vec[0]), Term::NIL)
                            .unwrap(),
                        2 => arc_process
                            .cons(
                                arc_process.integer(byte_vec[0]),
                                arc_process
                                    .cons(arc_process.integer(byte_vec[1]), Term::NIL)
                                    .unwrap(),
                            )
                            .unwrap(),
                        3 => arc_process
                            .cons(
                                arc_process.integer(byte_vec[0]),
                                arc_process
                                    .cons(
                                        arc_process.integer(byte_vec[1]),
                                        arc_process
                                            .cons(arc_process.integer(byte_vec[2]), Term::NIL)
                                            .unwrap(),
                                    )
                                    .unwrap(),
                            )
                            .unwrap(),
                        len => unimplemented!("len = {:?}", len),
                    }
                };

                let bitstring = arc_process.binary_from_bytes(&byte_vec).unwrap();

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
                        0 => Term::NIL,
                        1 => arc_process
                            .cons(arc_process.integer(byte_vec[0]), Term::NIL)
                            .unwrap(),
                        2 => arc_process
                            .cons(
                                arc_process.integer(byte_vec[0]),
                                arc_process
                                    .cons(arc_process.integer(byte_vec[1]), Term::NIL)
                                    .unwrap(),
                            )
                            .unwrap(),
                        3 => arc_process
                            .cons(
                                arc_process.integer(byte_vec[0]),
                                arc_process
                                    .cons(
                                        arc_process.integer(byte_vec[1]),
                                        arc_process
                                            .cons(arc_process.integer(byte_vec[2]), Term::NIL)
                                            .unwrap(),
                                    )
                                    .unwrap(),
                            )
                            .unwrap(),
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
                    let subbinary: SubBinary = bitstring.try_into().unwrap();

                    let byte_vec: Vec<u8> = subbinary.byte_iter().collect();

                    let mut bit_count_byte: u8 = 0;

                    for (i, bit) in subbinary.partial_byte_bit_iter().enumerate() {
                        bit_count_byte = bit_count_byte | (bit << (7 - i));
                    }

                    let bits_original_byte_vec = vec![bit_count_byte];
                    let bits_original = arc_process
                        .binary_from_bytes(&bits_original_byte_vec)
                        .unwrap();
                    let bits_subbinary = arc_process
                        .subbinary_from_original(
                            bits_original,
                            0,
                            0,
                            0,
                            subbinary.partial_byte_bit_len(),
                        )
                        .unwrap();

                    // not using an iterator because that would too closely match the code under
                    // test
                    let list = match byte_vec.len() {
                        0 => arc_process.cons(bits_subbinary, Term::NIL).unwrap(),
                        1 => arc_process
                            .cons(
                                arc_process.integer(byte_vec[0]),
                                arc_process.cons(bits_subbinary, Term::NIL).unwrap(),
                            )
                            .unwrap(),
                        2 => arc_process
                            .cons(
                                arc_process.integer(byte_vec[0]),
                                arc_process
                                    .cons(
                                        arc_process.integer(byte_vec[1]),
                                        arc_process.cons(bits_subbinary, Term::NIL).unwrap(),
                                    )
                                    .unwrap(),
                            )
                            .unwrap(),
                        3 => arc_process
                            .cons(
                                arc_process.integer(byte_vec[0]),
                                arc_process
                                    .cons(
                                        arc_process.integer(byte_vec[1]),
                                        arc_process
                                            .cons(
                                                arc_process.integer(byte_vec[2]),
                                                arc_process
                                                    .cons(bits_subbinary, Term::NIL)
                                                    .unwrap(),
                                            )
                                            .unwrap(),
                                    )
                                    .unwrap(),
                            )
                            .unwrap(),
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

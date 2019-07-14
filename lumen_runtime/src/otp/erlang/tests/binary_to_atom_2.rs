use super::*;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_binary(arc_process.clone()),
                    strategy::term::is_encoding(),
                ),
                |(binary, encoding)| {
                    prop_assert_eq!(
                        erlang::binary_to_atom_2(binary, encoding),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_without_encoding_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_binary(arc_process.clone()),
                    strategy::term::is_not_encoding(arc_process),
                ),
                |(binary, encoding)| {
                    prop_assert_eq!(
                        erlang::binary_to_atom_2(binary, encoding),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_utf8_binary_with_encoding_returns_atom_with_binary_name() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::is_utf8(arc_process.clone()),
                    strategy::term::is_encoding(),
                ),
                |(binary, encoding)| {
                    let byte_vec: Vec<u8> = match binary.to_typed_term().unwrap() {
                        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                            TypedTerm::HeapBinary(heap_binary) => heap_binary.as_bytes().to_vec(),
                            TypedTerm::SubBinary(subbinary) => subbinary.byte_iter().collect(),
                            unboxed_typed_term => {
                                panic!("unboxed_typed_term = {:?}", unboxed_typed_term)
                            }
                        },
                        typed_term => panic!("typed_term = {:?}", typed_term),
                    };

                    let s = std::str::from_utf8(&byte_vec).unwrap();

                    prop_assert_eq!(
                        erlang::binary_to_atom_2(binary, encoding),
                        Ok(atom_unchecked(s))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

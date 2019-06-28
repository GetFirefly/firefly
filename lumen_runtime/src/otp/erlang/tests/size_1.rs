use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_tuple_or_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone())
                    .prop_filter("Term must not be a tuple or bitstring", |term| {
                        !(term.is_tuple() || term.is_bitstring())
                    }),
                |term| {
                    prop_assert_eq!(erlang::size_1(term, &arc_process), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_returns_arity() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(0_usize..=3_usize).prop_flat_map(|size| {
                    (
                        Just(size),
                        strategy::term::tuple::intermediate(
                            strategy::term(arc_process.clone()),
                            (size..=size).into(),
                            arc_process.clone(),
                        ),
                    )
                }),
                |(size, term)| {
                    prop_assert_eq!(
                        erlang::size_1(term, &arc_process),
                        Ok(size.into_process(&arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_bitstring_is_byte_len() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_bitstring(arc_process.clone()), |term| {
                let byte_len = match term.tag() {
                    Boxed => {
                        let unboxed: &Term = term.unbox_reference();

                        match unboxed.tag() {
                            HeapBinary => {
                                let heap_binary: &heap::Binary = term.unbox_reference();

                                heap_binary.byte_len()
                            }
                            Subbinary => {
                                let subbinary: &sub::Binary = term.unbox_reference();

                                subbinary.byte_count
                            }
                            _ => unreachable!(),
                        }
                    }
                    _ => unreachable!(),
                };

                prop_assert_eq!(
                    erlang::size_1(term, &arc_process),
                    Ok(byte_len.into_process(&arc_process))
                );

                Ok(())
            })
            .unwrap();
    });
}

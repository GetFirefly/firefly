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
                    prop_assert_eq!(erlang::size_1(term, &arc_process), Err(badarg!().into()));

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
                        Ok(arc_process.integer(size))
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
                let full_byte_len = match term.to_typed_term().unwrap() {
                    TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                        TypedTerm::HeapBinary(heap_binary) => heap_binary.full_byte_len(),
                        TypedTerm::SubBinary(subbinary) => subbinary.full_byte_len(),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };

                prop_assert_eq!(
                    erlang::size_1(term, &arc_process),
                    Ok(arc_process.integer(full_byte_len))
                );

                Ok(())
            })
            .unwrap();
    });
}

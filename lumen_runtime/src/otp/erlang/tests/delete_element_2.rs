use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_tuple_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(tuple, index)| {
                    prop_assert_eq!(
                        erlang::delete_element_2(tuple, index, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_without_integer_between_1_and_the_length_inclusive_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &((
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
                    .prop_filter("Index either needs to not be an integer or not be an integer in the index range 1..=len", |(tuple, index)| {
                        let index_big_int_result: std::result::Result<BigInt, _> = index.try_into();

                        match index_big_int_result {
                            Ok(index_big_int) => {
                                let tuple_tuple: &Tuple = tuple.unbox_reference();
                                let min_index: BigInt = 1.into();
                                let max_index: BigInt = tuple_tuple.len().into();

                                !((min_index <= index_big_int) && (index_big_int <= max_index))
                            }
                            _ => true,
                        }
                    })),
                |(tuple, index)| {
                    prop_assert_eq!(
                        erlang::delete_element_2(tuple, index, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_with_integer_between_1_and_the_length_inclusive_returns_tuple_without_element() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(1_usize..=4_usize)
                    .prop_flat_map(|len| {
                        (
                            proptest::collection::vec(
                                strategy::term(arc_process.clone()),
                                len..=len,
                            ),
                            0..len,
                        )
                    })
                    .prop_map(|(element_vec, zero_based_index)| {
                        (
                            element_vec.clone(),
                            zero_based_index,
                            Term::slice_to_tuple(&element_vec, &arc_process),
                            (zero_based_index + 1).into_process(&arc_process),
                        )
                    }),
                |(mut element_vec, element_vec_index, tuple, index)| {
                    element_vec.remove(element_vec_index);

                    prop_assert_eq!(
                        erlang::delete_element_2(tuple, index, &arc_process),
                        Ok(Term::slice_to_tuple(&element_vec, &arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

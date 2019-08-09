use super::*;

#[test]
fn without_tuple_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term(arc_process),
                )
            }),
            |(arc_process, tuple, index, element)| {
                prop_assert_eq!(
                    erlang::setelement_3(index, tuple, element, &arc_process),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_without_valid_index_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple::without_index(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |((tuple, index), element)| {
                    prop_assert_eq!(
                        erlang::setelement_3(index, tuple, element, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_with_valid_index_returns_tuple_with_index_replaced() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple::with_index(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |((mut element_vec, element_vec_index, tuple, index), element)| {
                    element_vec[element_vec_index] = element;
                    let new_tuple = arc_process.tuple_from_slice(&element_vec).unwrap();

                    prop_assert_eq!(
                        erlang::setelement_3(index, tuple, element, &arc_process),
                        Ok(new_tuple)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

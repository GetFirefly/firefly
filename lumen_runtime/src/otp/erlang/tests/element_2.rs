use super::*;

#[test]
fn without_tuple_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term::is_integer(arc_process),
                )
            }),
            |(tuple, index)| {
                prop_assert_eq!(erlang::element_2(index, tuple), Err(badarg!().into()));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_without_integer_between_1_and_the_length_inclusive_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::tuple::without_index(arc_process.clone()),
                |(tuple, index)| {
                    prop_assert_eq!(
                        erlang::delete_element_2(index, tuple, &arc_process),
                        Err(badarg!().into())
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
                &strategy::term::tuple::with_index(arc_process.clone()),
                |(element_vec, element_vec_index, tuple, index)| {
                    prop_assert_eq!(
                        erlang::element_2(index, tuple),
                        Ok(element_vec[element_vec_index])
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

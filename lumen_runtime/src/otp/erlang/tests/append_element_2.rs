use super::*;

#[test]
fn without_tuple_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(tuple, element)| {
                    prop_assert_eq!(
                        erlang::append_element_2(tuple, element, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_returns_tuple_with_new_element_at_end() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(tuple, element)| {
                    let result = erlang::append_element_2(tuple, element, &arc_process);

                    prop_assert!(result.is_ok());

                    let appended_tuple = result.unwrap();

                    let appended_tuple_tuple_result: core::result::Result<Boxed<Tuple>, _> =
                        appended_tuple.try_into();

                    prop_assert!(appended_tuple_tuple_result.is_ok());

                    let appended_tuple_tuple = appended_tuple_tuple_result.unwrap();
                    let tuple_tuple: Boxed<Tuple> = tuple.try_into().unwrap();

                    prop_assert_eq!(appended_tuple_tuple.len(), tuple_tuple.len() + 1);
                    prop_assert_eq!(appended_tuple_tuple[tuple_tuple.len()], element);

                    Ok(())
                },
            )
            .unwrap();
    });
}

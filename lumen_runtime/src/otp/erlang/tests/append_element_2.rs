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
                        Err(badarg!())
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

                    prop_assert_eq!(appended_tuple.tag(), Boxed);

                    let unboxed_appended_tuple: &Term = appended_tuple.unbox_reference();

                    prop_assert_eq!(unboxed_appended_tuple.tag(), Arity);

                    let appended_tuple_tuple: &Tuple = appended_tuple.unbox_reference();
                    let tuple_tuple: &Tuple = tuple.unbox_reference();

                    prop_assert_eq!(appended_tuple_tuple.len(), tuple_tuple.len() + 1);
                    prop_assert_eq!(appended_tuple_tuple[tuple_tuple.len()], element);

                    Ok(())
                },
            )
            .unwrap();
    });
}

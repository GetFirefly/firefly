use super::*;

#[test]
fn without_tuple_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_tuple(arc_process.clone()),
                |tuple| {
                    prop_assert_eq!(erlang::tuple_to_list_1(tuple, &arc_process), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_returns_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &proptest::collection::vec(strategy::term(arc_process.clone()), 0..=3),
                |element_vec| {
                    let tuple = Term::slice_to_tuple(&element_vec, &arc_process);
                    let list = Term::slice_to_list(&element_vec, &arc_process);

                    prop_assert_eq!(erlang::tuple_to_list_1(tuple, &arc_process), Ok(list));

                    Ok(())
                },
            )
            .unwrap();
    });
}

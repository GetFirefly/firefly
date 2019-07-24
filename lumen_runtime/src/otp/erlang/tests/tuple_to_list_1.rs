use super::*;

#[test]
fn without_tuple_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::is_not_tuple(arc_process),
                )
            }),
            |(arc_process, tuple)| {
                prop_assert_eq!(
                    erlang::tuple_to_list_1(tuple, &arc_process),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_returns_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &proptest::collection::vec(strategy::term(arc_process.clone()), 0..=3),
                |element_vec| {
                    let tuple = arc_process.tuple_from_slice(&element_vec).unwrap();
                    let list = arc_process.list_from_slice(&element_vec).unwrap();

                    prop_assert_eq!(erlang::tuple_to_list_1(tuple, &arc_process), Ok(list));

                    Ok(())
                },
            )
            .unwrap();
    });
}

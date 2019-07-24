use super::*;

use proptest::strategy::Strategy;

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
                    erlang::tuple_size_1(tuple, &arc_process),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
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
                        Ok(arc_process.integer(size).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

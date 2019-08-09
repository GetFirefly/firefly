use super::*;

#[test]
fn without_tuple_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| strategy::term::is_not_tuple(arc_process)),
            |term| {
                prop_assert_eq!(erlang::is_tuple_1(term), false.into());

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::tuple(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_tuple_1(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}

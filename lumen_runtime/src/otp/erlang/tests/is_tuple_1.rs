use super::*;

#[test]
fn without_tuple_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_tuple(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_tuple_1(term), false.into());

                Ok(())
            })
            .unwrap();
    });
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

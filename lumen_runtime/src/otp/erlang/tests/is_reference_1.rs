use super::*;

#[test]
fn without_reference_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_reference(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_reference_1(term), false.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_tuple_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_reference(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_reference_1(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}

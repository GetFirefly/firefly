use super::*;

#[test]
fn without_map_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_map(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_map_1(term), false.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_map_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_map(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_map_1(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}

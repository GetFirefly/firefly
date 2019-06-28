use super::*;

#[test]
fn without_atom_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_atom(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_atom_1(term), false.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_atom_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::atom(), |term| {
            prop_assert_eq!(erlang::is_atom_1(term), true.into());

            Ok(())
        })
        .unwrap();
}

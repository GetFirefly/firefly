use super::*;

mod with_atom_name;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_atom(arc_process.clone()), |name| {
                prop_assert_eq!(erlang::unregister_1(name), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}

use super::*;

mod with_atom_class;

#[test]
fn without_atom_class_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    strategy::term::list::proper(arc_process.clone()),
                ),
                |(class, reason, stacktrace)| {
                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

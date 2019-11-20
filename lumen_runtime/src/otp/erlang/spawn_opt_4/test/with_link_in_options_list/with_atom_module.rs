use super::*;

mod with_atom_function;

#[test]
fn without_atom_function_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term::list::proper(arc_process.clone()),
                ),
                |(module, function, arguments)| {
                    prop_assert_eq!(
                        native(
                            &arc_process,
                            module,
                            function,
                            arguments,
                            options(&arc_process)
                        ),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

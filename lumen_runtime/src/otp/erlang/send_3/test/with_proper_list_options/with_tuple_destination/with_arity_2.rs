use super::*;

mod with_atom_name;

#[test]
fn without_atom_name_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(name, message, options)| {
                    let destination = arc_process
                        .tuple_from_slice(&[name, erlang::node_0::native()])
                        .unwrap();

                    prop_assert_eq!(
                        native(&arc_process, destination, message, options),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

use super::*;

mod with_atom_node;

#[test]
fn without_atom_node_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(name, node, message)| {
                    let destination = arc_process.tuple_from_slice(&[name, node]).unwrap();

                    prop_assert_eq!(
                        erlang::send_2(destination, message, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

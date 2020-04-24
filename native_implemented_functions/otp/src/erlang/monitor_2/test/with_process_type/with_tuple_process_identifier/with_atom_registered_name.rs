mod with_local_node;

use super::*;

#[test]
fn without_atom_node_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::is_not_atom(arc_process.clone()),
                ),
                |(registered_name, node)| {
                    let identifier = arc_process
                        .tuple_from_slice(&[registered_name, node])
                        .unwrap();

                    prop_assert_is_not_atom!(result(&arc_process, r#type(), identifier), node);

                    Ok(())
                },
            )
            .unwrap();
    });
}

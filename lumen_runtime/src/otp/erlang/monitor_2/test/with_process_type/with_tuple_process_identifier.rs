mod with_atom_registered_name;

use super::*;

#[test]
fn without_atom_registered_name_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_atom(arc_process.clone()),
                |registered_name| {
                    let identifier = arc_process
                        .tuple_from_slice(&[registered_name, node_0::native()])
                        .unwrap();

                    prop_assert_eq!(
                        native(&arc_process, r#type(), identifier),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

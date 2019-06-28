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
                    let destination = Term::slice_to_tuple(&[name, erlang::node_0()], &arc_process);

                    prop_assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

use super::*;

mod with_atom_name;

#[test]
fn without_atom_name_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term::pid_or_port(arc_process.clone()),
                ),
                |(name, pid_or_port)| {
                    prop_assert_eq!(
                        erlang::register_2(name, pid_or_port, arc_process.clone()),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

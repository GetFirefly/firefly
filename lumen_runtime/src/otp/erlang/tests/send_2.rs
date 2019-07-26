use super::*;

mod with_atom_destination;
mod with_local_pid_destination;
mod with_tuple_destination;

#[test]
fn without_atom_pid_or_tuple_destination_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::is_not_destination(arc_process.clone()),
                    strategy::term(arc_process),
                )
            }),
            |(arc_process, destination, message)| {
                prop_assert_eq!(
                    erlang::send_2(destination, message, &arc_process),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
}

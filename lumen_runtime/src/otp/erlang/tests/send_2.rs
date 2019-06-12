use super::*;

use proptest::strategy::Strategy;

mod with_atom_destination;
mod with_local_pid_destination;
mod with_tuple_destination;

#[test]
fn without_atom_pid_or_tuple_destination_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()).prop_filter(
                        "Destination must not be an atom, pid, or tuple",
                        |destination| {
                            !(destination.is_atom()
                                || destination.is_pid()
                                || destination.is_tuple())
                        },
                    ),
                    strategy::term(arc_process.clone()),
                ),
                |(destination, message)| {
                    prop_assert_eq!(
                        erlang::send_2(destination, message, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

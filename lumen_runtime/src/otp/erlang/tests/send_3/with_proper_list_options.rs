use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

mod empty;
mod non_empty;

mod with_atom_destination;
mod with_local_pid_destination;
mod with_tuple_destination;

#[test]
fn without_atom_pid_or_tuple_destination_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_destination(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(destination, message, options)| {
                    prop_assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn valid_options(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    let noconnect = atom_unchecked("noconnect");
    let nosuspend = atom_unchecked("nosuspend");

    prop_oneof![
        Just(empty::OPTIONS),
        Just(arc_process.list_from_slice(&[noconnect]).unwrap()),
        Just(arc_process.list_from_slice(&[nosuspend]).unwrap()),
        Just(
            arc_process
                .list_from_slice(&[noconnect, nosuspend])
                .unwrap()
        )
    ]
    .boxed()
}

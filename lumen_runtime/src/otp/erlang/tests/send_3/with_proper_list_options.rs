use super::*;

use proptest::prop_oneof;

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
                    is_not_destination(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(destination, message, options)| {
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

fn valid_options(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let noconnect = Term::str_to_atom("noconnect", DoNotCare).unwrap();
    let nosuspend = Term::str_to_atom("nosuspend", DoNotCare).unwrap();

    prop_oneof![
        Just(empty::OPTIONS),
        Just(Term::slice_to_list(&[noconnect], &arc_process)),
        Just(Term::slice_to_list(&[nosuspend], &arc_process)),
        Just(Term::slice_to_list(&[noconnect, nosuspend], &arc_process))
    ]
    .boxed()
}

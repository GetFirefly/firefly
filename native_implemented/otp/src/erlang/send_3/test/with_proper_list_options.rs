use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

mod non_empty;

mod with_atom_destination;
mod with_local_pid_destination;
mod with_tuple_destination;

#[test]
fn without_atom_pid_or_tuple_destination_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_destination(arc_process.clone()),
                strategy::term(arc_process.clone()),
                valid_options(arc_process),
            )
        },
        |(arc_process, destination, message, options)| {
            prop_assert_badarg!(
                    result(&arc_process, destination, message, options),
                    format!("destination ({}) is not registered_name (atom), {{registered_name, node}}, or pid", destination)
                );

            Ok(())
        },
    );
}

fn valid_options(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let noconnect = Atom::str_to_term("noconnect");
    let nosuspend = Atom::str_to_term("nosuspend");

    prop_oneof![
        Just(Term::NIL),
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

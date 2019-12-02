use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

mod with_atom_destination;
mod with_local_pid_destination;

#[test]
fn without_atom_or_pid_destination_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term::is_not_send_after_destination(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(milliseconds, destination, message)| {
                    let time = arc_process.integer(milliseconds).unwrap();
                    let options = options(&arc_process);

                    prop_assert_eq!(
                        native(arc_process.clone(), time, destination, message, options),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn milliseconds() -> BoxedStrategy<Milliseconds> {
    prop_oneof![
        Just(timer::at_once_milliseconds()),
        Just(timer::soon_milliseconds()),
        Just(timer::later_milliseconds()),
        Just(timer::long_term_milliseconds())
    ]
    .boxed()
}

use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

mod with_atom_destination;
mod with_local_pid_destination;

#[test]
fn without_atom_pid_or_tuple_destination_errors_badarg() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term::is_not_send_after_destination(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, milliseconds, destination, message)| {
            let time = arc_process.integer(milliseconds).unwrap();
            let options = options(&arc_process);

            prop_assert_badarg!(
                native(arc_process.clone(), time, destination, message, options),
                "supported option is {:abs, bool}"
            );

            Ok(())
        },
    );
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

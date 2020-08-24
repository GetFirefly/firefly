// `with_empty_list_options` in integration tests
// `with_link_and_monitor_in_options_list` in integration tests
// `with_link_in_options_list` in integration tests
// `with_monitor_in_options_list` in integration tests

use super::*;

#[test]
fn without_proper_list_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_function(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, function, tail)| {
            let options = arc_process.improper_list_from_slice(&[atom!("link")], tail);

            prop_assert_badarg!(result(&arc_process, function, options), "improper list");

            Ok(())
        },
    );
}

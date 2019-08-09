use super::*;

use std::thread;
use std::time::Duration;

mod with_proper_list_options;

#[test]
fn without_proper_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::non_negative(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                ),
                |(time, message, options)| {
                    let destination = arc_process.pid_term();

                    prop_assert_eq!(
                        erlang::start_timer_4(
                            time,
                            destination,
                            message,
                            options,
                            arc_process.clone()
                        ),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

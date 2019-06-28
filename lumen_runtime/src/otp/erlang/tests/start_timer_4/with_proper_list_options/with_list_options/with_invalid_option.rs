use super::*;

mod with_small_integer_time;

// BigInt is not tested because it would take too long and would always count as `long_term` for the
// super short soon and later wheel sizes used for `cfg(test)`

#[test]
fn without_non_negative_integer_time_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_non_negative_integer(arc_process.clone()),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                ),
                |(time, message)| {
                    let destination = arc_process.pid;
                    let options = options(&arc_process);

                    prop_assert_eq!(
                        erlang::start_timer_4(
                            time,
                            destination,
                            message,
                            options,
                            arc_process.clone()
                        ),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn options(process: &Process) -> Term {
    Term::cons(
        Term::str_to_atom("invalid", DoNotCare).unwrap(),
        Term::EMPTY_LIST,
        process,
    )
}

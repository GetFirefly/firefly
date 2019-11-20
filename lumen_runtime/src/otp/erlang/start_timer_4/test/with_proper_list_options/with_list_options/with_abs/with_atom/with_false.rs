use super::*;

mod with_small_integer_time;

// BigInt is not tested because it would take too long and would always count as `long_term` for the
// super shot soon and later wheel sizes used for `cfg(test)`

#[test]
fn without_non_negative_integer_time_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_non_negative_integer(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(time, message)| {
                    let destination = arc_process.pid_term();
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

fn options(process: &Process) -> Term {
    super::options(false.into(), process)
}

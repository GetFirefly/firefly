use super::*;

use proptest::strategy::Strategy;

mod with_small_integer_time;

// BigInt is not tested because it would take too long and would always count as `long_term` for the
// super short soon and later wheel sizes used for `cfg(test)`

#[test]
fn without_non_negative_integer_time_error_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_non_negative_integer(arc_process.clone()),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                    options(arc_process.clone()),
                ),
                |(time, message, options)| {
                    let destination = arc_process.pid;

                    prop_assert_eq!(
                        erlang::send_after_4(
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

fn options(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process.clone())
        .prop_map(move |value| super::options(value, &arc_process))
        .boxed()
}

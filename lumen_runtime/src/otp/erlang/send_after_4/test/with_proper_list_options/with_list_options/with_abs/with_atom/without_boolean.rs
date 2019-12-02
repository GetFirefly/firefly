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
                    strategy::term(arc_process.clone()),
                    options(arc_process.clone()),
                ),
                |(time, message, options)| {
                    let destination = arc_process.pid_term();

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

fn options(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process.clone())
        .prop_map(move |value| super::options(value, &arc_process))
        .boxed()
}

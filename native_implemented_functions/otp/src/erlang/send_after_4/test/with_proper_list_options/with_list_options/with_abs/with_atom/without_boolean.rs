use super::*;

mod with_small_integer_time;

// BigInt is not tested because it would take too long and would always count as `long_term` for the
// super short soon and later wheel sizes used for `cfg(test)`

#[test]
fn without_non_negative_integer_time_error_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_non_negative_integer(arc_process.clone()),
                strategy::term(arc_process.clone()),
                abs_value(arc_process.clone()),
            )
        },
        |(arc_process, time, message, abs_value)| {
            let destination = arc_process.pid_term();
            let options = options(abs_value, &arc_process);

            prop_assert_is_not_boolean!(
                result(arc_process.clone(), time, destination, message, options),
                "abs value",
                abs_value
            );

            Ok(())
        },
    );
}

fn abs_value(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process.clone())
}

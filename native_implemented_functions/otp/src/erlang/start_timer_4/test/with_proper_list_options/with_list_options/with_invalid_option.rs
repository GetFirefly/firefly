use super::*;

mod with_small_integer_time;

// BigInt is not tested because it would take too long and would always count as `long_term` for the
// super short soon and later wheel sizes used for `cfg(test)`

#[test]
fn without_non_negative_integer_time_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_non_negative_integer(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, time, message)| {
            let destination = arc_process.pid_term();
            let options = options(&arc_process);

            prop_assert_badarg!(
                result(arc_process.clone(), time, destination, message, options),
                "supported option is {:abs, bool}"
            );

            Ok(())
        },
    );
}

fn options(process: &Process) -> Term {
    process
        .cons(Atom::str_to_term("invalid"), Term::NIL)
        .unwrap()
}

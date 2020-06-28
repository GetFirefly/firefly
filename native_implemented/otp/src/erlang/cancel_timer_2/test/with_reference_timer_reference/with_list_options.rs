use super::*;

mod with_async_false;
mod with_async_true;
mod without_async;

#[test]
fn with_invalid_option() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |option| {
                let timer_reference = arc_process.next_reference().unwrap();
                let options = arc_process.cons(option, Term::NIL).unwrap();

                prop_assert_badarg!(
                    result(&arc_process, timer_reference, options),
                    format!("supported options are {{:async, bool}} or {{:info, bool}}")
                );

                Ok(())
            })
            .unwrap();
    })
}

fn async_option(value: bool, process: &Process) -> Term {
    option("async", value, process)
}

fn info_option(value: bool, process: &Process) -> Term {
    option("info", value, process)
}

fn option(key: &str, value: bool, process: &Process) -> Term {
    process
        .tuple_from_slice(&[Atom::str_to_term(key), value.into()])
        .unwrap()
}

fn with_timer_in_same_thread_without_timeout_returns_ok_and_does_not_send_timeout_message(
    options: fn(&Process) -> Term,
) {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        let start_time_in_milliseconds = freeze_timeout();
        freeze_at_timeout(start_time_in_milliseconds + milliseconds / 2 + 1);

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        // again before timeout
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
    })
}

fn with_timer_in_same_thread_with_timeout_returns_ok_after_timeout_message_was_sent(
    options: fn(&Process) -> Term,
) {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        let start_time_in_milliseconds = freeze_timeout();
        freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

        let timeout_message = timeout_message(timer_reference, message, process);

        assert_has_message!(process, timeout_message);

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        // again
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
    })
}

fn with_info_false_without_timer_returns_ok(options: fn(&Process) -> Term) {
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
    });
}

fn without_info_without_local_reference_errors_badarg(
    source_file: &'static str,
    options: fn(&Process) -> Term,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_local_reference(arc_process.clone()),
            )
        },
        |(arc_process, timer_reference)| {
            prop_assert_badarg!(
                result(&arc_process, timer_reference, options(&arc_process)),
                format!(
                    "timer_reference ({}) is not a local reference",
                    timer_reference
                )
            );

            Ok(())
        },
    );
}

fn returns_false(options: fn(&Process) -> Term) {
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(false.into())
        );
    });
}

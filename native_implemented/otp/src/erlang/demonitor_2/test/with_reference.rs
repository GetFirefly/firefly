mod with_flush_and_info_options;
mod with_flush_option;
mod with_info_option;
mod without_options;

use super::*;

#[test]
fn without_proper_list_for_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_reference(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, reference, tail)| {
            let options = arc_process
                .improper_list_from_slice(&[atom!("flush")], tail)
                .unwrap();

            prop_assert_badarg!(result(&arc_process, reference, options), "improper list");

            Ok(())
        },
    );
}

#[test]
fn with_unknown_option_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_reference(arc_process.clone()),
                unknown_option(arc_process.clone()),
            )
        },
        |(arc_process, reference, option)| {
            let options = arc_process.list_from_slice(&[option]).unwrap();

            prop_assert_badarg!(
                result(&arc_process, reference, options),
                "supported options are :flush or :info"
            );

            Ok(())
        },
    );
}

fn prevents_future_messages(options: fn(&Process) -> Term) {
    with_process_arc(|monitoring_arc_process| {
        let monitored_arc_process = process::child(&monitoring_arc_process);
        let monitored_pid_term = monitored_arc_process.pid_term();

        let monitor_reference =
            monitor_2::result(&monitoring_arc_process, r#type(), monitored_pid_term).unwrap();

        let reason = Atom::str_to_term("normal");
        let tag = Atom::str_to_term("DOWN");

        assert!(!has_message(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid_term, reason])
                .unwrap()
        ));

        assert_eq!(
            result(
                &monitoring_arc_process,
                monitor_reference,
                options(&monitoring_arc_process)
            ),
            Ok(true.into())
        );

        exit_when_run(&monitored_arc_process, reason);

        assert!(scheduler::run_through(&monitored_arc_process));

        assert!(monitored_arc_process.is_exiting());
        assert!(!monitoring_arc_process.is_exiting());

        assert!(!has_message(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid_term, reason])
                .unwrap()
        ));
    });
}

fn unknown_option(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter("Option cannot be flush or info", |option| {
            match option.decode().unwrap() {
                TypedTerm::Atom(atom) => match atom.name() {
                    "flush" | "info" => false,
                    _ => true,
                },
                _ => true,
            }
        })
        .boxed()
}

fn r#type() -> Term {
    Atom::str_to_term("process")
}

fn with_monitor_returns_true(options: fn(&Process) -> Term) {
    with_process_arc(|monitoring_arc_process| {
        let monitored_arc_process = process::child(&monitoring_arc_process);

        let monitor_reference = monitor_2::result(
            &monitoring_arc_process,
            r#type(),
            monitored_arc_process.pid_term(),
        )
        .unwrap();

        let monitored_monitor_count_before = monitor_count(&monitored_arc_process);
        let monitoring_monitored_count_before = monitored_count(&monitoring_arc_process);

        assert_eq!(
            result(
                &monitoring_arc_process,
                monitor_reference,
                options(&monitoring_arc_process)
            ),
            Ok(true.into())
        );

        let monitored_monitor_count_after = monitor_count(&monitored_arc_process);
        let monitoring_monitored_count_after = monitored_count(&monitoring_arc_process);

        assert_eq!(
            monitored_monitor_count_after,
            monitored_monitor_count_before - 1
        );
        assert_eq!(
            monitoring_monitored_count_after,
            monitoring_monitored_count_before - 1
        );
    });
}

fn with_info_option_without_monitor_returns_false(options: fn(&Process) -> Term) {
    with_process_arc(|monitoring_arc_process| {
        let reference = monitoring_arc_process.next_reference().unwrap();

        assert_eq!(
            result(
                &monitoring_arc_process,
                reference,
                options(&monitoring_arc_process)
            ),
            Ok(false.into())
        )
    });
}

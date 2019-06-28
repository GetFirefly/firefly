use super::*;

mod with_different_process;

#[test]
fn without_process_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let destination = process::identifier::local::next();

                    prop_assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_process_adds_process_message_to_mailbox_and_returns_ok() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let destination = arc_process.pid;

                    prop_assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
                    );

                    prop_assert!(has_process_message(&arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}

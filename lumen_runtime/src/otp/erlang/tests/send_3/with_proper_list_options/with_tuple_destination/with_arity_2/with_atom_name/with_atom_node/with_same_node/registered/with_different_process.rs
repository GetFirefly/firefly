use super::*;

#[test]
fn without_locked_adds_process_message_to_mailbox_and_returns_ok() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let different_arc_process = process::local::test(&arc_process);
                    let name = registered_name();

                    prop_assert_eq!(
                        erlang::register_2(name, different_arc_process.pid, arc_process.clone()),
                        Ok(true.into())
                    );

                    let destination = Term::slice_to_tuple(&[name, erlang::node_0()], &arc_process);

                    prop_assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
                    );

                    prop_assert!(has_process_message(&different_arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_locked_adds_heap_message_to_mailbox_and_returns_ok() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let different_arc_process = process::local::test(&arc_process);
                    let name = registered_name();

                    prop_assert_eq!(
                        erlang::register_2(name, different_arc_process.pid, arc_process.clone()),
                        Ok(true.into())
                    );

                    let _different_process_heap_lock = different_arc_process.heap.lock().unwrap();

                    let destination = Term::slice_to_tuple(&[name, erlang::node_0()], &arc_process);

                    prop_assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
                    );

                    prop_assert!(has_heap_message(&different_arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}

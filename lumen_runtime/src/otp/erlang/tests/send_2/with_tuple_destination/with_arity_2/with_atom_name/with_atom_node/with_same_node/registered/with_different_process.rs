use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_locked_adds_heap_message_to_mailbox_and_returns_message() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::heap_fragment_safe(arc_process),
                )
            }),
            |(arc_process, message)| {
                let name = registered_name();
                let different_arc_process = process::test(&arc_process);

                prop_assert_eq!(
                    erlang::register_2(name, different_arc_process.pid_term(), arc_process.clone()),
                    Ok(true.into())
                );

                let _different_process_heap_lock = different_arc_process.acquire_heap();

                let destination = arc_process
                    .tuple_from_slice(&[name, erlang::node_0()])
                    .unwrap();

                prop_assert_eq!(
                    erlang::send_2(destination, message, &arc_process),
                    Ok(message)
                );

                prop_assert!(has_heap_message(&different_arc_process, message));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn without_locked_adds_process_message_to_mailbox_and_returns_message() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::heap_fragment_safe(arc_process),
                )
            }),
            |(arc_process, message)| {
                let name = registered_name();
                let different_process = process::test(&arc_process);

                prop_assert_eq!(
                    erlang::register_2(name, different_process.pid_term(), arc_process.clone()),
                    Ok(true.into())
                );

                let destination = arc_process
                    .tuple_from_slice(&[name, erlang::node_0()])
                    .unwrap();

                prop_assert_eq!(
                    erlang::send_2(destination, message, &arc_process),
                    Ok(message)
                );

                prop_assert!(has_process_message(&different_process, message));

                Ok(())
            },
        )
        .unwrap();
}

use super::*;

#[test]
fn with_non_existent_pid_errors_noproc() {
    with_process(|process| {
        let link_count_before = link_count(process);

        assert_eq!(
            result(process, Pid::next_term()),
            Err(error(
                Atom::str_to_term("noproc"),
                None,
                Trace::capture(),
                Some(anyhow!("Test").into())
            )
            .into())
        );

        assert_eq!(link_count(process), link_count_before);
    });
}

// `with_existing_unlinked_pid_links_to_process` in integration tests
// `with_existing_linked_pid_returns_true` in integration tests
// `when_a_linked_process_exits_normal_the_process_does_not_exit` in integration tests
// `when_a_linked_process_exits_unexpected_the_process_does_not_exit` in integration tests
// `when_the_process_exits_unexpected_linked_processes_exit_too` in integration tests

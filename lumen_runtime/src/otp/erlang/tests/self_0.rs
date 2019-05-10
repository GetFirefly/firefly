use super::*;

#[test]
fn returns_process_pid() {
    with_process(|process| {
        assert_eq!(erlang::self_0(&process), process.pid);
    });
}

use crate::erlang::self_0::result;
use crate::test::with_process;

#[test]
fn returns_process_pid() {
    with_process(|process| {
        assert_eq!(result(&process), process.pid_term());
    });
}

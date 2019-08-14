use crate::otp::erlang::self_0::native;
use crate::scheduler::with_process;

#[test]
fn returns_process_pid() {
    with_process(|process| {
        assert_eq!(native(&process), process.pid_term());
    });
}

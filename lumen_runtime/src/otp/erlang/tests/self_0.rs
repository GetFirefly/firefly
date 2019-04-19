use super::*;

use crate::process;

#[test]
fn returns_process_pid() {
    let process = process::local::new();

    assert_eq!(erlang::self_0(&process), process.pid);
}

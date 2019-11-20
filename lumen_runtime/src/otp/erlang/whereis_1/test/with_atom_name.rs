use super::*;

use crate::scheduler::with_process;

#[test]
fn without_registered_name_returns_undefined() {
    let name = registered_name();

    with_process(|process| {
        assert_eq!(native(process, name), Ok(Atom::str_to_term("undefined")));
    });
}

#[test]
fn with_registered_name_returns_pid() {
    with_process_arc(|process_arc| {
        let name = registered_name();
        let pid_or_port = process_arc.pid();

        assert_eq!(
            erlang::register_2::native(process_arc.clone(), name, pid_or_port.into()),
            Ok(true.into())
        );

        assert_eq!(native(&process_arc, name), Ok(pid_or_port.into()));
    })
}

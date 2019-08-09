use super::*;

#[test]
fn without_registered_name_returns_undefined() {
    let name = registered_name();

    assert_eq!(erlang::whereis_1(name), Ok(atom_unchecked("undefined")));
}

#[test]
fn with_registered_name_returns_pid() {
    with_process_arc(|process_arc| {
        let name = registered_name();
        let pid_or_port = unsafe { process_arc.pid().as_term() };

        assert_eq!(
            erlang::register_2(name, pid_or_port, process_arc.clone()),
            Ok(true.into())
        );

        assert_eq!(erlang::whereis_1(name), Ok(pid_or_port));
    })
}

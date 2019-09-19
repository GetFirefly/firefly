use super::*;

#[test]
fn without_registered_name_returns_undefined() {
    let name = registered_name();

    assert_eq!(native(name), Ok(atom_unchecked("undefined")));
}

#[test]
fn with_registered_name_returns_pid() {
    with_process_arc(|process_arc| {
        let name = registered_name();
        let pid_or_port = unsafe { process_arc.pid().as_term() };

        assert_eq!(
            erlang::register_2::native(process_arc.clone(), name, pid_or_port),
            Ok(true.into())
        );

        assert_eq!(native(name), Ok(pid_or_port));
    })
}

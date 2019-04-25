use super::*;

use crate::registry::Registered;

#[test]
fn without_process() {
    with_process_arc(|process_arc| {
        let pid_or_port = process::identifier::local::next();

        assert_badarg!(erlang::register_2(
            registered_name(),
            pid_or_port,
            process_arc
        ));
    });
}

#[test]
fn with_same_process() {
    with_process_arc(|process_arc| {
        let name = registered_name();
        let pid_or_port = process_arc.pid;

        assert_eq!(
            erlang::register_2(name, pid_or_port, process_arc.clone()),
            Ok(true.into())
        );
        assert_eq!(*process_arc.registered_name.read().unwrap(), Some(name));
        assert_eq!(
            registry::RW_LOCK_REGISTERED_BY_NAME
                .read()
                .unwrap()
                .get(&name),
            Some(&Registered::Process(process_arc))
        );
    });
}

#[test]
fn with_different_process() {
    with_process_arc(|process_arc| {
        let name = registered_name();

        let another_process_arc = process::local::new();
        let pid_or_port = another_process_arc.pid;

        assert_eq!(
            erlang::register_2(name, pid_or_port, process_arc),
            Ok(true.into())
        );
        assert_eq!(
            *another_process_arc.registered_name.read().unwrap(),
            Some(name)
        );
        assert_eq!(
            registry::RW_LOCK_REGISTERED_BY_NAME
                .read()
                .unwrap()
                .get(&name),
            Some(&Registered::Process(another_process_arc))
        );
    });
}

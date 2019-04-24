use super::*;

#[test]
fn with_same_process_returns_true() {
    with_process_arc(|process_arc| {
        let name = registered_name();
        let pid_or_port = process_arc.pid;

        assert_eq!(
            erlang::register_2(name, pid_or_port, process_arc.clone()),
            Ok(true.into())
        );

        assert_eq!(*process_arc.registered_name.lock().unwrap(), Some(name));
        assert_eq!(
            registry::RW_LOCK_REGISTERED_BY_NAME
                .read()
                .unwrap()
                .get(&name),
            Some(&Registered::Process(process_arc.clone()))
        );

        assert_eq!(erlang::unregister_1(name), Ok(true.into()));

        assert_eq!(*process_arc.registered_name.lock().unwrap(), None);
        assert_eq!(
            registry::RW_LOCK_REGISTERED_BY_NAME
                .read()
                .unwrap()
                .get(&name),
            None
        );
    })
}

#[test]
fn with_different_process_returns_true() {
    with_process_arc(|process_arc| {
        let name = registered_name();

        let another_process_arc = process::local::new();
        let pid_or_port = another_process_arc.pid;

        assert_eq!(
            erlang::register_2(name, pid_or_port, process_arc.clone()),
            Ok(true.into())
        );

        assert_eq!(*process_arc.registered_name.lock().unwrap(), None);
        assert_eq!(
            *another_process_arc.registered_name.lock().unwrap(),
            Some(name)
        );
        assert_eq!(
            registry::RW_LOCK_REGISTERED_BY_NAME
                .read()
                .unwrap()
                .get(&name),
            Some(&Registered::Process(another_process_arc.clone()))
        );

        assert_eq!(erlang::unregister_1(name), Ok(true.into()));

        assert_eq!(*another_process_arc.registered_name.lock().unwrap(), None);
        assert_eq!(
            registry::RW_LOCK_REGISTERED_BY_NAME
                .read()
                .unwrap()
                .get(&name),
            None
        );
    })
}

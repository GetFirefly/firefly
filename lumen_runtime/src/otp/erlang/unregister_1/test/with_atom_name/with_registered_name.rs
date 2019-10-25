use super::*;

#[test]
fn with_same_process_returns_true() {
    with_process_arc(|process_arc| {
        let name = registered_name();
        let name_atom: Atom = name.try_into().unwrap();
        let pid_or_port = unsafe { process_arc.pid().decode() };

        assert_eq!(
            erlang::register_2::native(process_arc.clone(), name, pid_or_port),
            Ok(true.into())
        );

        assert_eq!(*process_arc.registered_name.read(), Some(name_atom));

        let name_atom: Atom = name.try_into().unwrap();

        assert_eq!(
            registry::atom_to_process(&name_atom),
            Some(process_arc.clone())
        );

        assert_eq!(native(name), Ok(true.into()));

        assert_eq!(*process_arc.registered_name.read(), None);
        assert_eq!(registry::atom_to_process(&name_atom), None);
    })
}

#[test]
fn with_different_process_returns_true() {
    with_process_arc(|process_arc| {
        let name = registered_name();
        let name_atom: Atom = name.try_into().unwrap();

        let another_process_arc = process::test(&process_arc);
        let pid_or_port = unsafe { another_process_arc.pid().decode() };

        assert_eq!(
            erlang::register_2::native(process_arc.clone(), name, pid_or_port),
            Ok(true.into())
        );

        assert_eq!(*process_arc.registered_name.read(), None);
        assert_eq!(*another_process_arc.registered_name.read(), Some(name_atom));

        let name_atom: Atom = name.try_into().unwrap();

        assert_eq!(
            registry::atom_to_process(&name_atom),
            Some(another_process_arc.clone())
        );

        assert_eq!(native(name), Ok(true.into()));

        assert_eq!(*another_process_arc.registered_name.read(), None);
        assert_eq!(registry::atom_to_process(&name_atom), None);
    })
}

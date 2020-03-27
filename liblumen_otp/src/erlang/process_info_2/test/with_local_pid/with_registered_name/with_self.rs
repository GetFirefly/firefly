use super::*;

#[test]
fn without_registered_returns_empty_list() {
    with_process_arc(|unregistered_process_arc| {
        assert_eq!(
            native(
                &unregistered_process_arc,
                unregistered_process_arc.pid_term(),
                item()
            ),
            Ok(Term::NIL)
        );
    });
}

#[test]
fn with_registered_returns_empty_list() {
    with_process_arc(|registered_process_arc| {
        let registered_name = registered_name();
        let registered_name_atom: Atom = registered_name.try_into().unwrap();

        assert!(registry::put_atom_to_process(
            registered_name_atom,
            registered_process_arc.clone()
        ));

        assert_eq!(
            native(
                &registered_process_arc,
                registered_process_arc.pid_term(),
                item()
            ),
            Ok(registered_process_arc
                .tuple_from_slice(&[item(), registered_name])
                .unwrap())
        );
    });
}

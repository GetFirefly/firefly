use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::term::Atom;

use crate::test::registered_name;
use crate::{process, registry};

#[test]
fn without_registered_returns_empty_list() {
    with_process_arc(|parent_process_arc| {
        let unregistred_arc_process = process::test(&parent_process_arc);

        assert_eq!(
            native(
                &parent_process_arc,
                unregistred_arc_process.pid_term(),
                item()
            ),
            Ok(Term::NIL)
        );
    });
}

#[test]
fn with_registered_returns_empty_list() {
    with_process_arc(|parent_process_arc| {
        let registered_process_arc = process::test(&parent_process_arc);
        let registered_name = registered_name();
        let registered_name_atom: Atom = registered_name.try_into().unwrap();

        assert!(registry::put_atom_to_process(
            registered_name_atom,
            registered_process_arc.clone()
        ));

        assert_eq!(
            native(
                &parent_process_arc,
                registered_process_arc.pid_term(),
                item()
            ),
            Ok(parent_process_arc
                .tuple_from_slice(&[item(), registered_name])
                .unwrap())
        );
    });
}

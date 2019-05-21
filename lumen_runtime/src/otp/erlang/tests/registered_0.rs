use super::*;

// because registry is global and tests are concurrent, there is no way to test for completely
// empty registry

#[test]
fn includes_registered_process_name() {
    with_process_arc(|process_arc| {
        let name = Term::str_to_atom("registered_process_name", DoNotCare).unwrap();

        let before = erlang::registered_0(&process_arc);

        match before.tag() {
            EmptyList => (),
            List => {
                let before_cons: &Cons = unsafe { before.as_ref_cons_unchecked() };

                assert!(!before_cons.contains(name));
            }
            tag => panic!("Wrong tag ({:?})", tag),
        }

        assert_eq!(
            erlang::register_2(name, process_arc.pid, process_arc.clone()),
            Ok(true.into())
        );

        let after = erlang::registered_0(&process_arc);

        assert_eq!(after.tag(), List);

        let after_cons: &Cons = unsafe { after.as_ref_cons_unchecked() };

        assert!(after_cons.contains(name));
    });
}

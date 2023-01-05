// because registry is global and tests are concurrent, there is no way to test for completely
// empty registry

use firefly_rt::term::{Atom, Term};

use crate::erlang;
use crate::erlang::registered_0::result;
use crate::test::with_process_arc;

#[test]
fn includes_registered_process_name() {
    with_process_arc(|process_arc| {
        let name: Term = Atom::str_to_term("registered_process_name").into();

        let before = result(&process_arc).unwrap();

        match before {
            Term::Nil => (),
            Term::Cons(before_cons) => {
                assert!(!before_cons.contains(name));
            }
            typed_term => panic!("Wrong TypedTerm ({:?})", typed_term),
        }

        assert_eq!(
            erlang::register_2::result(process_arc.clone(), name, process_arc.pid_term()),
            Ok(true.into())
        );

        let after = result(&process_arc).unwrap();

        match after {
            Term::Cons(after_cons) => assert!(after_cons.contains(name)),
            typed_term => panic!("Wrong TypedTerm ({:?})", typed_term),
        }
    });
}

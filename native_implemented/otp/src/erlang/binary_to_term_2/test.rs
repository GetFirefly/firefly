mod with_safe;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::{Atom, Term};

use crate::erlang;
use crate::erlang::binary_to_term_2::result;
use crate::test::strategy;

#[test]
fn with_used_with_binary_returns_how_many_bytes_were_consumed_along_with_term() {
    // <<131,100,0,5,"hello","world">>
    let byte_vec = vec![
        131, 100, 0, 5, 104, 101, 108, 108, 111, 119, 111, 114, 108, 100,
    ];

    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::containing_bytes(byte_vec.clone(), arc_process.clone()),
            )
        },
        |(arc_process, binary)| {
            let options = options(&arc_process);
            let term = Atom::str_to_term("hello");

            prop_assert_eq!(
                result(&arc_process, binary, options),
                Ok(arc_process
                    .tuple_from_slice(&[term, arc_process.integer(9).unwrap()])
                    .unwrap())
            );

            // Using only `used` portion of binary returns the same result
            let tuple = result(&arc_process, binary, options).unwrap();
            let used_term =
                erlang::element_2::result(arc_process.integer(2).unwrap(), tuple).unwrap();
            let split_binary_tuple =
                erlang::split_binary_2::result(&arc_process, binary, used_term).unwrap();
            let prefix =
                erlang::element_2::result(arc_process.integer(1).unwrap(), split_binary_tuple)
                    .unwrap();

            prop_assert_eq!(result(&arc_process, prefix, options), Ok(tuple));

            // Without used returns only term

            prop_assert_eq!(result(&arc_process, binary, Term::NIL), Ok(term));

            Ok(())
        },
    );

    fn options(process: &Process) -> Term {
        process.cons(Atom::str_to_term("used"), Term::NIL).unwrap()
    }
}

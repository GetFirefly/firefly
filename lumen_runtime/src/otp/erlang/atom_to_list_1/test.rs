use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::{Atom, Term};

use crate::otp::erlang::atom_to_list_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_atom(arc_process.clone()), |atom| {
                prop_assert_is_not_atom!(native(&arc_process, atom), atom);

                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore]
fn with_atom_returns_chars_in_list() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), any::<String>())
                .prop_map(|(arc_process, string)| (arc_process, Atom::str_to_term(&string), string))
        },
        |(arc_process, atom, string)| {
            let codepoint_terms: Vec<Term> = string
                .chars()
                .map(|c| arc_process.integer(c).unwrap())
                .collect();

            prop_assert_eq!(
                native(&arc_process, atom),
                Ok(arc_process.list_from_slice(&codepoint_terms).unwrap())
            );

            Ok(())
        },
    );
}

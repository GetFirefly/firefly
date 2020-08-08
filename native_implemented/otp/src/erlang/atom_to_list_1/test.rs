use proptest::test_runner::{Config, TestRunner};

use crate::erlang::atom_to_list_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_atom(arc_process.clone()), |atom| {
                prop_assert_is_not_atom!(result(&arc_process, atom), atom);

                Ok(())
            })
            .unwrap();
    });
}

// `with_atom_returns_chars_in_list` in integration tests

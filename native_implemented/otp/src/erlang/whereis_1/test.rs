mod with_atom_name;

use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;
use crate::erlang::whereis_1::result;
use crate::test::with_process_arc;
use crate::test::{registered_name, strategy};

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_atom(arc_process.clone()), |name| {
                prop_assert_is_not_atom!(result(name), name);

                Ok(())
            })
            .unwrap();
    });
}

mod with_atom_class;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::raise_3::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_atom_class_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    strategy::term::list::proper(arc_process.clone()),
                ),
                |(class, reason, stacktrace)| {
                    prop_assert_badarg!(
                        native(class, reason, stacktrace),
                        format!("class ({}) is not an atom", class)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

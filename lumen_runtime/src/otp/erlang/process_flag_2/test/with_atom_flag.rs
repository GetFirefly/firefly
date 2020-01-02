mod with_trap_exit_flag;

use super::*;

use std::convert::TryInto;

use proptest::strategy::BoxedStrategy;

use liblumen_alloc::erts::term::prelude::{Atom, Term};

#[test]
fn without_supported_flag_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(unsupported_flag_atom(), strategy::term(arc_process.clone())),
                |(flag, value)| {
                    prop_assert_badarg!(native(&arc_process, flag, value), "supported flags are error_handler, max_heap_size, message_queue_data, min_bin_vheap_size, min_heap_size, priority, save_calls, sensitive, and trap_exit");

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn unsupported_flag_atom() -> BoxedStrategy<Term> {
    strategy::term::atom()
        .prop_filter("Cannot be a supported flag name", |atom| {
            let atom_atom: Atom = (*atom).try_into().unwrap();

            match atom_atom.name() {
                "trap_exit" => false,
                _ => true,
            }
        })
        .boxed()
}

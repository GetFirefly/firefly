mod with_atom_left;
mod with_big_integer_left;
mod with_empty_list_left;
mod with_external_pid_left;
mod with_float_left;
mod with_heap_binary_left;
mod with_list_left;
mod with_local_pid_left;
mod with_local_reference_left;
mod with_map_left;
mod with_small_integer_left;
mod with_subbinary_left;
mod with_tuple_left;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::is_less_than_2::result;
use crate::test::{external_arc_node, strategy};
use crate::test::{with_process, with_process_arc};

#[test]
fn with_same_left_and_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |operand| {
                prop_assert_eq!(result(operand, operand), false.into());

                Ok(())
            })
            .unwrap();
    });
}

fn is_less_than<L, R>(left: L, right: R, expected: bool)
where
    L: FnOnce(&Process) -> Term,
    R: FnOnce(Term, &Process) -> Term,
{
    with_process(|process| {
        let left = left(&process);
        let right = right(left, &process);

        assert_eq!(result(left, right), expected.into());
    });
}

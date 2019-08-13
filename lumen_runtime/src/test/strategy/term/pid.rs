use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::{make_pid, Pid, Term};
use liblumen_alloc::erts::ProcessControlBlock;

pub mod external;

pub fn external(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    let external_pid_arc_process = arc_process.clone();

    (external::node_id(), number(), serial())
        .prop_map(move |(node, number, serial)| {
            external_pid_arc_process
                .external_pid_with_node_id(node, number, serial)
                .unwrap()
        })
        .boxed()
}

pub fn local() -> BoxedStrategy<Term> {
    (number(), serial())
        .prop_map(|(number, serial)| make_pid(number, serial).unwrap())
        .boxed()
}

pub fn number() -> BoxedStrategy<usize> {
    (0..=Pid::NUMBER_MAX).boxed()
}

pub fn serial() -> BoxedStrategy<usize> {
    (0..=Pid::SERIAL_MAX).boxed()
}

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use firefly_rt::process::Process;
use firefly_rt::term::{Pid, Term};

use crate::test::strategy::node;

pub fn external(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let external_pid_arc_process = arc_process.clone();

    (node::external(), number(), serial())
        .prop_map(move |(arc_node, number, serial)| {
            external_pid_arc_process
                .external_pid(arc_node, number, serial)
                .unwrap()
        })
        .boxed()
}

pub fn local() -> BoxedStrategy<Term> {
    (number(), serial())
        .prop_map(|(number, serial)| Pid::make_term(number, serial).unwrap())
        .boxed()
}

pub fn number() -> BoxedStrategy<usize> {
    (0..=Pid::NUMBER_MAX).boxed()
}

pub fn serial() -> BoxedStrategy<usize> {
    (0..=Pid::SERIAL_MAX).boxed()
}

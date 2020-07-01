use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::closure::Creator;
use liblumen_alloc::erts::term::prelude::{ExternalPid, Pid};

use crate::test::strategy::node;
use crate::test::strategy::term::pid;

pub fn external() -> BoxedStrategy<Creator> {
    (node::external(), pid::number(), pid::serial())
        .prop_map(|(arc_node, number, serial)| ExternalPid::new(arc_node, number, serial).unwrap())
        .prop_map(|external_pid| Creator::External(external_pid))
        .boxed()
}

pub fn local() -> BoxedStrategy<Creator> {
    (pid::number(), pid::serial())
        .prop_map(|(number, serial)| Pid::new(number, serial).unwrap())
        .prop_map(|local_pid| Creator::Local(local_pid))
        .boxed()
}

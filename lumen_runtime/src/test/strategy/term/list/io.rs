use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::test::strategy::term::{is_binary, is_byte};
use crate::test::strategy::{DEPTH, MAX_LEN};

pub fn non_recursive_element(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![is_byte(arc_process.clone()), is_binary(arc_process.clone())].boxed()
}

pub fn recursive_element(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    non_recursive_element(arc_process.clone())
        .prop_recursive(
            DEPTH,
            (MAX_LEN * (DEPTH as usize + 1)) as u32,
            MAX_LEN as u32,
            move |element_strategy| {
                let elements_strategy = proptest::collection::vec(element_strategy, 0..=MAX_LEN);

                (
                    Just(arc_process.clone()),
                    elements_strategy,
                    tail(arc_process.clone()),
                )
                    .prop_map(|(arc_process, elements, tail)| {
                        arc_process
                            .improper_list_from_slice(&elements, tail)
                            .unwrap()
                    })
                    .boxed()
            },
        )
        .boxed()
}

pub fn recursive_elements(arc_process: Arc<Process>) -> BoxedStrategy<Vec<Term>> {
    proptest::collection::vec(recursive_element(arc_process), 0..=MAX_LEN).boxed()
}

pub fn root(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (
        Just(arc_process.clone()),
        recursive_elements(arc_process.clone()),
        tail(arc_process),
    )
        .prop_map(|(arc_process, elements, tail)| {
            arc_process
                .improper_list_from_slice(&elements, tail)
                .unwrap()
        })
        .boxed()
}

pub fn tail(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![is_binary(arc_process.clone()), Just(Term::NIL)].boxed()
}

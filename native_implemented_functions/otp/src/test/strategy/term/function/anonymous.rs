mod creator;
mod without_native;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Creator;
use liblumen_alloc::erts::term::prelude::Term;

use crate::test;

pub fn creator() -> BoxedStrategy<Creator> {
    prop_oneof![creator::local(), creator::external()].boxed()
}

pub fn index() -> BoxedStrategy<u32> {
    // A `u32`, but must be encodable as an `i32` for INTEGER_EXT
    (0..=(std::i32::MAX as u32)).boxed()
}

pub fn old_unique() -> BoxedStrategy<u32> {
    // A `u32`, but must be encodable as an `i32` for INTEGER_EXT
    (0..=(std::i32::MAX as u32)).boxed()
}

pub fn unique() -> BoxedStrategy<[u8; 16]> {
    proptest::collection::vec(any::<u8>(), 16..=16)
        .prop_map(|byte_vec| {
            let bytes: &[u8] = &byte_vec;
            let byte_array: [u8; 16] = bytes.try_into().unwrap();

            byte_array
        })
        .boxed()
}

pub fn with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    match arity {
        0 => prop_oneof![
            Just(test::anonymous_0::anonymous_closure(&arc_process).unwrap()),
            without_native::with_arity(arc_process, arity)
        ]
        .boxed(),
        1 => prop_oneof![
            Just(test::anonymous_1::anonymous_closure(&arc_process).unwrap()),
            without_native::with_arity(arc_process, arity)
        ]
        .boxed(),
        _ => without_native::with_arity(arc_process, arity),
    }
}

pub fn without_native(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (
        super::module_atom(),
        index(),
        old_unique(),
        unique(),
        super::arity_u8(),
        creator(),
    )
        .prop_map(move |(module, index, old_unique, unique, arity, creator)| {
            arc_process
                .anonymous_closure_with_env_from_slice(
                    module,
                    index,
                    old_unique,
                    unique,
                    arity,
                    None,
                    creator,
                    &[],
                )
                .unwrap()
        })
        .boxed()
}

pub fn with_native(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (0..=1)
        .prop_map(move |arity| {
            // MUST be functions in symbol table passed to `runtime::test::once` in
            // `test::process::init`.
            match arity {
                0 => test::anonymous_0::anonymous_closure(&arc_process).unwrap(),
                1 => test::anonymous_1::anonymous_closure(&arc_process).unwrap(),
                _ => unreachable!("arity {}", arity),
            }
        })
        .boxed()
}

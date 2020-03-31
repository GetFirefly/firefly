mod creator;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Creator;
use liblumen_alloc::erts::term::prelude::Term;

use crate::test::strategy::term;

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
    (
        super::module_atom(),
        index(),
        old_unique(),
        unique(),
        super::option_debuggable_code(),
        creator(),
        proptest::collection::vec(term(arc_process.clone()), 0..2),
    )
        .prop_map(
            move |(module, index, old_unique, unique, option_debuggable_code, creator, env_vec)| {
                let option_code = option_debuggable_code.map(|debuggable_code| debuggable_code.0);

                if let Some(code) = option_code {
                    crate::runtime::code::anonymous::insert(
                        module, index, old_unique, unique, arity, code,
                    );
                }

                arc_process
                    .anonymous_closure_with_env_from_slice(
                        module,
                        index,
                        old_unique,
                        unique,
                        arity,
                        option_code,
                        creator,
                        &env_vec,
                    )
                    .unwrap()
            },
        )
        .boxed()
}

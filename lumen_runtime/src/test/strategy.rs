use std::ops::RangeInclusive;
use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::process;

pub mod base;
pub mod byte_vec;
pub mod module_function_arity;
pub mod node;
pub mod size_range;
pub mod term;

pub const NON_EMPTY_RANGE_INCLUSIVE: RangeInclusive<usize> = 1..=MAX_LEN;
pub const NON_EXISTENT_ATOM_PREFIX: &str = "non_existent";

pub fn atom() -> BoxedStrategy<Atom> {
    any::<String>()
        .prop_filter("Reserved for existing/safe atom tests", |s| {
            !s.starts_with(NON_EXISTENT_ATOM_PREFIX)
        })
        .prop_map(|s| Atom::try_from_str(&s).unwrap())
        .boxed()
}

pub fn bits_to_bytes(bits: usize) -> usize {
    (bits + 7) / 8
}

pub fn byte_vec() -> BoxedStrategy<Vec<u8>> {
    byte_vec::with_size_range(RANGE_INCLUSIVE.into())
}

pub fn process() -> BoxedStrategy<Arc<Process>> {
    Just(process::test_init())
        .prop_flat_map(|init_arc_process| {
            // generated in a prop_flat_map instead of prop_map so that a new process is generated
            // for each test so that a prior run's register doesn't interfere
            Just(process::test(&init_arc_process))
        })
        .boxed()
}

pub fn term(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let container_arc_process = arc_process.clone();

    term::leaf(RANGE_INCLUSIVE, arc_process)
        .prop_recursive(
            DEPTH,
            (MAX_LEN * (DEPTH as usize + 1)) as u32,
            MAX_LEN as u32,
            move |element| {
                term::container(
                    element,
                    RANGE_INCLUSIVE.clone().into(),
                    container_arc_process.clone(),
                )
            },
        )
        .boxed()
}

pub fn without_integer_start_with_integer_length(
    arc_process: Arc<Process>,
) -> (
    impl Strategy<Value = Arc<Process>>,
    impl Strategy<Value = Term>,
    impl Strategy<Value = Term>,
    impl Strategy<Value = Term>,
) {
    (
        Just(arc_process.clone()),
        term::is_bitstring(arc_process.clone()),
        term::is_not_integer(arc_process.clone()),
        term::is_integer(arc_process.clone()),
    )
}

pub fn without_integer_start_without_integer_length(
    arc_process: Arc<Process>,
) -> (
    impl Strategy<Value = Arc<Process>>,
    impl Strategy<Value = Term>,
    impl Strategy<Value = Term>,
    impl Strategy<Value = Term>,
) {
    (
        Just(arc_process.clone()),
        term::is_bitstring(arc_process.clone()),
        term::is_not_integer(arc_process.clone()),
        term::is_not_integer(arc_process.clone()),
    )
}

const DEPTH: u32 = 3;
const MAX_LEN: usize = 3;
const RANGE_INCLUSIVE: RangeInclusive<usize> = 0..=MAX_LEN;

pub fn size_range() -> SizeRange {
    RANGE_INCLUSIVE.clone().into()
}

use std::ptr::NonNull;
use std::sync::Arc;

use num_bigint::BigInt;

use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Just, Strategy};
use firefly_alloc::gc::GcBox;

use firefly_rt::process::Process;
use firefly_rt::term::{Term, Tuple};

pub fn intermediate(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| arc_process.tuple_term_from_term_slice(&vec).unwrap())
        .boxed()
}

pub fn with_index(arc_process: Arc<Process>) -> BoxedStrategy<(Vec<Term>, usize, Term, Term)> {
    (Just(arc_process.clone()), 1_usize..=4_usize)
        .prop_flat_map(|(arc_process, len)| {
            (
                Just(arc_process.clone()),
                proptest::collection::vec(super::super::term(arc_process), len..=len),
                0..len,
            )
        })
        .prop_map(|(arc_process, element_vec, zero_based_index)| {
            (
                element_vec.clone(),
                zero_based_index,
                arc_process.tuple_term_from_term_slice(&element_vec).unwrap(),
                arc_process.integer(zero_based_index + 1).unwrap(),
            )
        })
        .boxed()
}

pub fn non_empty(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    intermediate(
        super::super::term(arc_process.clone()),
        (1..=3).into(),
        arc_process,
    )
    .boxed()
}

pub fn without_index(arc_process: Arc<Process>) -> BoxedStrategy<(Term, Term)> {
    (non_empty(arc_process.clone()), super::super::term(arc_process.clone()))
        .prop_filter("Index either needs to not be an integer or not be an integer in the index range 1..=len", |(tuple, index)| {
            let index_gc_box_big_int_result: Result<GcBox<BigInt>, _> = (*index).try_into();

            match index_gc_box_big_int_result {
                Ok(index_gc_box_big_int) => {
                    let non_null_tuple: NonNull<Tuple> = (*tuple).try_into().unwrap();
                    let min_index: BigInt = 1.into();
                    let index_big_int = index_gc_box_big_int.as_ref();
                    let max_index: BigInt = non_null_tuple.as_ref().len().into();

                    !((min_index <= *./graindex_big_int) && (index_big_int <= *max_index))
                }
                _ => true,
            }
        }).boxed()
}

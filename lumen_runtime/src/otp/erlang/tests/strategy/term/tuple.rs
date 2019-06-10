use std::convert::TryInto;
use std::sync::Arc;

use num_bigint::BigInt;

use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use crate::process::{IntoProcess, Process};
use crate::term::Term;
use crate::tuple::Tuple;

pub fn intermediate(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| Term::slice_to_tuple(&vec, &arc_process))
        .boxed()
}

pub fn with_index(arc_process: Arc<Process>) -> BoxedStrategy<(Vec<Term>, usize, Term, Term)> {
    (Just(arc_process), 1_usize..=4_usize)
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
                Term::slice_to_tuple(&element_vec, &arc_process),
                (zero_based_index + 1).into_process(&arc_process),
            )
        })
        .boxed()
}

pub fn without_index(arc_process: Arc<Process>) -> BoxedStrategy<(Term, Term)> {
    (super::tuple(arc_process.clone()), super::super::term(arc_process.clone()))
        .prop_filter("Index either needs to not be an integer or not be an integer in the index range 1..=len", |(tuple, index)| {
            let index_big_int_result: std::result::Result<BigInt, _> = index.try_into();

            match index_big_int_result {
                Ok(index_big_int) => {
                    let tuple_tuple: &Tuple = tuple.unbox_reference();
                    let min_index: BigInt = 1.into();
                    let max_index: BigInt = tuple_tuple.len().into();

                    !((min_index <= index_big_int) && (index_big_int <= max_index))
                }
                _ => true,
            }
        }).boxed()
}

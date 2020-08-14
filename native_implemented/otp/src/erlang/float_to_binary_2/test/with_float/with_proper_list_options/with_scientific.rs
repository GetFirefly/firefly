use super::*;

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::binary_to_string::binary_to_string;

// `with_20_digits_is_the_same_as_float_to_binary_1` in integration tests
// `returns_binary_with_coefficient_e_exponent` in integration tests

#[test]
fn always_includes_e() {
    run!(strategy, |(arc_process, float, options)| {
        let result = result(&arc_process, float, options);

        prop_assert!(result.is_ok());

        let binary = result.unwrap();
        let string: String = binary_to_string(binary).unwrap();

        prop_assert!(string.contains('e'));

        Ok(())
    },);
}

#[test]
fn always_includes_sign_of_exponent() {
    run!(strategy, |(arc_process, float, options)| {
        let result = result(&arc_process, float, options);

        prop_assert!(result.is_ok());

        let binary = result.unwrap();
        let string: String = binary_to_string(binary).unwrap();
        let part_vec: Vec<&str> = string.splitn(2, 'e').collect();

        prop_assert_eq!(part_vec.len(), 2);

        let sign = part_vec[1].chars().nth(0).unwrap();

        prop_assert!(sign == '+' || sign == '-');

        Ok(())
    },);
}

#[test]
fn exponent_is_at_least_2_digits() {
    run!(strategy, |(arc_process, float, options)| {
        let result = result(&arc_process, float, options);

        prop_assert!(result.is_ok());

        let binary = result.unwrap();
        let string: String = binary_to_string(binary).unwrap();
        let part_vec: Vec<&str> = string.splitn(2, 'e').collect();

        prop_assert_eq!(part_vec.len(), 2);

        prop_assert!(2 <= part_vec[1].chars().skip(1).count());

        Ok(())
    },);
}

fn digits(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (Just(arc_process.clone()), 0..=249)
        .prop_map(|(arc_process, u)| arc_process.integer(u).unwrap())
        .boxed()
}

fn strategy(arc_process: Arc<Process>) -> impl Strategy<Value = (Arc<Process>, Term, Term)> {
    (
        Just(arc_process.clone()),
        super::strategy::term::float(arc_process.clone()),
        (Just(arc_process.clone()), digits(arc_process.clone())).prop_map(
            |(arc_process, digits)| {
                arc_process
                    .list_from_slice(&[arc_process.tuple_from_slice(&[tag(), digits]).unwrap()])
                    .unwrap()
            },
        ),
    )
}

fn tag() -> Term {
    Atom::str_to_term("scientific")
}

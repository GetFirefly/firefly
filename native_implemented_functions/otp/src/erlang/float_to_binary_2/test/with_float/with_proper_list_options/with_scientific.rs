use super::*;

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::float_to_binary_1;
use crate::runtime::binary_to_string::binary_to_string;

#[test]
fn with_20_digits_is_the_same_as_float_to_binary_1() {
    with_process_arc(|arc_process| {
        let digits = arc_process.integer(20).unwrap();
        let options = arc_process
            .list_from_slice(&[arc_process.tuple_from_slice(&[tag(), digits]).unwrap()])
            .unwrap();

        let zero = arc_process.float(0.0).unwrap();

        assert_eq!(
            native(&arc_process, zero, options).unwrap(),
            float_to_binary_1::native(&arc_process, zero).unwrap()
        );

        let one_tenth = arc_process.float(0.1).unwrap();

        assert_eq!(
            native(&arc_process, one_tenth, options).unwrap(),
            float_to_binary_1::native(&arc_process, one_tenth).unwrap()
        );
    });
}

#[test]
fn returns_binary_with_coefficient_e_exponent() {
    with_process_arc(|arc_process| {
        let float = arc_process.float(1234567890.0987654321).unwrap();

        assert_eq!(
            native(&arc_process, float, options(&arc_process, 0)),
            Ok(arc_process.binary_from_str("1e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 1)),
            Ok(arc_process.binary_from_str("1.2e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 2)),
            Ok(arc_process.binary_from_str("1.23e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 3)),
            Ok(arc_process.binary_from_str("1.235e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 4)),
            Ok(arc_process.binary_from_str("1.2346e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 5)),
            Ok(arc_process.binary_from_str("1.23457e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 6)),
            Ok(arc_process.binary_from_str("1.234568e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 7)),
            Ok(arc_process.binary_from_str("1.2345679e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 8)),
            Ok(arc_process.binary_from_str("1.23456789e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 9)),
            Ok(arc_process.binary_from_str("1.234567890e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 10)),
            Ok(arc_process.binary_from_str("1.2345678901e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 11)),
            Ok(arc_process.binary_from_str("1.23456789010e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 12)),
            Ok(arc_process.binary_from_str("1.234567890099e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 13)),
            Ok(arc_process.binary_from_str("1.2345678900988e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 14)),
            Ok(arc_process.binary_from_str("1.23456789009877e+09").unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 15)),
            Ok(arc_process
                .binary_from_str("1.234567890098765e+09")
                .unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 16)),
            Ok(arc_process
                .binary_from_str("1.2345678900987654e+09")
                .unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 17)),
            Ok(arc_process
                .binary_from_str("1.23456789009876537e+09")
                .unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 18)),
            Ok(arc_process
                .binary_from_str("1.234567890098765373e+09")
                .unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 19)),
            Ok(arc_process
                .binary_from_str("1.2345678900987653732e+09")
                .unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 20)),
            Ok(arc_process
                .binary_from_str("1.23456789009876537323e+09")
                .unwrap())
        );
        assert_eq!(
            native(&arc_process, float, options(&arc_process, 21)),
            Ok(arc_process
                .binary_from_str("1.234567890098765373230e+09")
                .unwrap())
        );
    });
}

#[test]
fn always_includes_e() {
    run!(strategy, |(arc_process, float, options)| {
        let result = native(&arc_process, float, options);

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
        let result = native(&arc_process, float, options);

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
        let result = native(&arc_process, float, options);

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

fn options(process: &Process, digits: u8) -> Term {
    process
        .list_from_slice(&[process
            .tuple_from_slice(&[tag(), process.integer(digits).unwrap()])
            .unwrap()])
        .unwrap()
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

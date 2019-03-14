use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn with_atom_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(atom_term, from_unit_term, to_unit_term, &mut process),
        process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(Term::EMPTY_LIST, from_unit_term, to_unit_term, &mut process),
        process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list_term = list_term(&mut process);
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(list_term, from_unit_term, to_unit_term, &mut process),
        process
    );
}

mod with_small_integer {
    use super::*;

    use num_traits::Num;

    #[test]
    fn without_valid_units_returns_bad_argument() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        let small_integer_term: Term = 0.into_process(&mut process);
        let valid_unit_term =
            Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
        let invalid_unit_term = Term::str_to_atom("s", Existence::DoNotCare, &mut process).unwrap();

        assert_bad_argument!(
            erlang::convert_time_unit(
                small_integer_term,
                valid_unit_term,
                invalid_unit_term,
                &mut process,
            ),
            process
        );

        assert_bad_argument!(
            erlang::convert_time_unit(
                small_integer_term,
                invalid_unit_term,
                valid_unit_term,
                &mut process,
            ),
            process
        );
    }

    #[test]
    fn with_valid_units_returns_converted_value() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        let small_integer_term: Term = 1_000_000_000.into_process(&mut process);

        assert_eq!(small_integer_term.tag(), Tag::SmallInteger);

        // (Hertz, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                2_usize.into_process(&mut process),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(2_000_000_000.into_process(&mut process)),
            process
        );
        // (Hertz, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(500_000_000.into_process(&mut process)),
            process
        );
        // (Hertz, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Hertz, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("500_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Hertz, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Hertz, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Hertz, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );

        // (Second, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Second, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000_usize.into_process(&mut process)),
            process
        );
        // (Second, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Second, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Second, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Second, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Second, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );

        // (Millisecond, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5_000_000.into_process(&mut process)),
            process
        );
        // (Millisecond, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process)),
            process
        );
        // (Millisecond, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Millisecond, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Millisecond, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Millisecond, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Millisecond, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );

        // (Microsecond, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5_000_000.into_process(&mut process)),
            process
        );
        // (Microsecond, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process)),
            process
        );
        // (Microsecond, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Microsecond, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Microsecond, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Microsecond, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Microsecond, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );

        // (Nanosecond, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5.into_process(&mut process)),
            process
        );
        // (Nanosecond, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1.into_process(&mut process)),
            process
        );
        // (Nanosecond, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000.into_process(&mut process)),
            process
        );
        // (Nanosecond, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process)),
            process
        );
        // (Nanosecond, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Nanosecond, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000.into_process(&mut process)),
            process
        );
        // (Nanosecond, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000.into_process(&mut process)),
            process
        );

        // (Native, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5_000_000.into_process(&mut process)),
            process
        );
        // (Native, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process)),
            process
        );
        // (Native, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Native, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Native, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Native, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Native, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );

        // (Native, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5_000_000.into_process(&mut process)),
            process
        );
        // (Native, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process)),
            process
        );
        // (Native, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Native, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Native, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Native, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Native, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                small_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
    }
}

mod with_big_integer {
    use super::*;

    use num_traits::Num;

    #[test]
    fn without_valid_units_returns_bad_argument() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        let big_integer_term: Term =
            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process);

        assert_eq!(big_integer_term.tag(), Tag::Boxed);

        let valid_unit_term =
            Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
        let invalid_unit_term = Term::str_to_atom("s", Existence::DoNotCare, &mut process).unwrap();

        assert_bad_argument!(
            erlang::convert_time_unit(
                big_integer_term,
                valid_unit_term,
                invalid_unit_term,
                &mut process,
            ),
            process
        );

        assert_bad_argument!(
            erlang::convert_time_unit(
                big_integer_term,
                invalid_unit_term,
                valid_unit_term,
                &mut process,
            ),
            process
        );
    }

    #[test]
    fn with_valid_units_returns_converted_value() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        let big_integer_term: Term =
            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process);

        assert_eq!(big_integer_term.tag(), Tag::Boxed);

        // (Hertz, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                2_usize.into_process(&mut process),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("2_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Hertz, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Hertz, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Hertz, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Hertz, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Hertz, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Hertz, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );

        // (Second, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("5_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Second, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Second, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Second, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Second, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Second, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Second, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );

        // (Millisecond, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Millisecond, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Millisecond, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Millisecond, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Millisecond, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Millisecond, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Millisecond, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );

        // (Microsecond, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Microsecond, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Microsecond, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Microsecond, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Microsecond, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Microsecond, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Microsecond, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );

        // (Nanosecond, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Nanosecond, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process)),
            process
        );
        // (Nanosecond, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Nanosecond, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Nanosecond, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Nanosecond, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Nanosecond, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );

        // (Native, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Native, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Native, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Native, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Native, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Native, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Native, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );

        // (Native, Hertz)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Native, Second)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process)),
            process
        );
        // (Native, Millisecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Native, Microsecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Native, Nanosecond)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Native, Native)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
        // (Native, PerformanceCounter)
        assert_eq_in_process!(
            erlang::convert_time_unit(
                big_integer_term,
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            ),
            process
        );
    }
}

#[test]
fn with_float_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let float_term = 1.0.into_process(&mut process);
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(float_term, from_unit_term, to_unit_term, &mut process),
        process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let local_pid_term = Term::local_pid(0, 0).unwrap();
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(local_pid_term, from_unit_term, to_unit_term, &mut process),
        process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(
            external_pid_term,
            from_unit_term,
            to_unit_term,
            &mut process
        ),
        process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(tuple_term, from_unit_term, to_unit_term, &mut process),
        process
    );
}

#[test]
fn with_heap_binary_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[1], &mut process);
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(heap_binary_term, from_unit_term, to_unit_term, &mut process),
        process
    );
}

#[test]
fn with_subbinary_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term = Term::slice_to_binary(&[0, 1], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 0, &mut process);
    let from_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
    let to_unit_term = Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::convert_time_unit(subbinary_term, from_unit_term, to_unit_term, &mut process),
        process
    );
}

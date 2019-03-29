use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    errors_badarg(|mut process| list_term(&mut process));
}

mod with_small_integer {
    use super::*;

    use num_traits::Num;

    #[test]
    fn without_valid_units_errors_badarg() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        let small_integer_term: Term = 0.into_process(&mut process);
        let valid_unit_term = Term::str_to_atom("native", DoNotCare).unwrap();
        let invalid_unit_term = Term::str_to_atom("s", DoNotCare).unwrap();

        assert_badarg!(erlang::convert_time_unit_3(
            small_integer_term,
            valid_unit_term,
            invalid_unit_term,
            &mut process,
        ));

        assert_badarg!(erlang::convert_time_unit_3(
            small_integer_term,
            invalid_unit_term,
            valid_unit_term,
            &mut process,
        ));
    }

    #[test]
    fn with_valid_units_returns_converted_value() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        let small_integer_term: Term = 1_000_000_000.into_process(&mut process);

        assert_eq!(small_integer_term.tag(), SmallInteger);

        // (Hertz, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                2_usize.into_process(&mut process),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(2_000_000_000.into_process(&mut process))
        );
        // (Hertz, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(500_000_000.into_process(&mut process))
        );
        // (Hertz, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Hertz, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("500_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Hertz, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Hertz, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Hertz, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );

        // (Second, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Second, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000_usize.into_process(&mut process))
        );
        // (Second, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Second, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Second, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Second, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Second, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );

        // (Millisecond, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5_000_000.into_process(&mut process))
        );
        // (Millisecond, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process))
        );
        // (Millisecond, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Millisecond, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Millisecond, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Millisecond, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Millisecond, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );

        // (Microsecond, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5_000_000.into_process(&mut process))
        );
        // (Microsecond, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process))
        );
        // (Microsecond, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Microsecond, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Microsecond, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Microsecond, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Microsecond, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );

        // (Nanosecond, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5.into_process(&mut process))
        );
        // (Nanosecond, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1.into_process(&mut process))
        );
        // (Nanosecond, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000.into_process(&mut process))
        );
        // (Nanosecond, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process))
        );
        // (Nanosecond, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Nanosecond, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000.into_process(&mut process))
        );
        // (Nanosecond, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000.into_process(&mut process))
        );

        // (Native, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5_000_000.into_process(&mut process))
        );
        // (Native, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process))
        );
        // (Native, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Native, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Native, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Native, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Native, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );

        // (Native, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(5_000_000.into_process(&mut process))
        );
        // (Native, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000.into_process(&mut process))
        );
        // (Native, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Native, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Native, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Native, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Native, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                small_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
    }
}

mod with_big_integer {
    use super::*;

    use num_traits::Num;

    #[test]
    fn without_valid_units_errors_badarg() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        let big_integer_term: Term =
            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process);

        assert_eq!(big_integer_term.tag(), Boxed);

        let valid_unit_term = Term::str_to_atom("native", DoNotCare).unwrap();
        let invalid_unit_term = Term::str_to_atom("s", DoNotCare).unwrap();

        assert_badarg!(erlang::convert_time_unit_3(
            big_integer_term,
            valid_unit_term,
            invalid_unit_term,
            &mut process,
        ));

        assert_badarg!(erlang::convert_time_unit_3(
            big_integer_term,
            invalid_unit_term,
            valid_unit_term,
            &mut process,
        ));
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

        assert_eq!(big_integer_term.tag(), Boxed);

        // (Hertz, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                2_usize.into_process(&mut process),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("2_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Hertz, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Hertz, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Hertz, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Hertz, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Hertz, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Hertz, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                2_usize.into_process(&mut process),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );

        // (Second, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("5_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Second, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Second, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Second, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Second, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Second, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Second, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("second", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );

        // (Millisecond, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Millisecond, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Millisecond, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Millisecond, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Millisecond, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Millisecond, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Millisecond, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );

        // (Microsecond, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Microsecond, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Microsecond, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Microsecond, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Microsecond, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Microsecond, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Microsecond, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );

        // (Nanosecond, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Nanosecond, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(1_000_000_000.into_process(&mut process))
        );
        // (Nanosecond, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Nanosecond, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Nanosecond, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Nanosecond, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Nanosecond, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );

        // (Native, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Native, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Native, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Native, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Native, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Native, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Native, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );

        // (Native, Hertz)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                5_usize.into_process(&mut process),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Native, Second)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("second", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                .unwrap()
                .into_process(&mut process))
        );
        // (Native, Millisecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Native, Microsecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Native, Nanosecond)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Native, Native)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("native", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
        // (Native, PerformanceCounter)
        assert_eq!(
            erlang::convert_time_unit_3(
                big_integer_term,
                Term::str_to_atom("native", DoNotCare).unwrap(),
                Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                &mut process
            ),
            Ok(
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&mut process)
            )
        );
    }
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_binary(&[1], &mut process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|mut process| {
        let original = Term::slice_to_binary(&[0, 1], &mut process);
        Term::subbinary(original, 1, 0, 1, 0, &mut process)
    });
}

fn errors_badarg<F>(time: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let from_unit = Term::str_to_atom("native", DoNotCare).unwrap();
        let to_unit = Term::str_to_atom("native", DoNotCare).unwrap();

        erlang::convert_time_unit_3(time(&mut process), from_unit, to_unit, &mut process)
    });
}

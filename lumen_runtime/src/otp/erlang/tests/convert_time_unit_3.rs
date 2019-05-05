use super::*;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    errors_badarg(|process| list_term(&process));
}

mod with_small_integer {
    use super::*;

    use num_traits::Num;

    #[test]
    fn without_valid_units_errors_badarg() {
        with_process(|process| {
            let small_integer_term: Term = 0.into_process(&process);
            let valid_unit_term = Term::str_to_atom("native", DoNotCare).unwrap();
            let invalid_unit_term = Term::str_to_atom("s", DoNotCare).unwrap();

            assert_badarg!(erlang::convert_time_unit_3(
                small_integer_term,
                valid_unit_term,
                invalid_unit_term,
                &process,
            ));

            assert_badarg!(erlang::convert_time_unit_3(
                small_integer_term,
                invalid_unit_term,
                valid_unit_term,
                &process,
            ));
        });
    }

    #[test]
    fn with_valid_units_returns_converted_value() {
        with_process(|process| {
            let small_integer_term: Term = 1_000_000_000.into_process(&process);

            assert_eq!(small_integer_term.tag(), SmallInteger);

            // (Hertz, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    2_usize.into_process(&process),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(2_000_000_000.into_process(&process))
            );
            // (Hertz, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(500_000_000.into_process(&process))
            );
            // (Hertz, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Hertz, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("500_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Hertz, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Hertz, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Hertz, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );

            // (Second, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("5_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Second, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000_usize.into_process(&process))
            );
            // (Second, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Second, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Second, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Second, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Second, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );

            // (Millisecond, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(5_000_000.into_process(&process))
            );
            // (Millisecond, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000.into_process(&process))
            );
            // (Millisecond, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Millisecond, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Millisecond, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Millisecond, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Millisecond, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );

            // (Microsecond, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(5_000_000.into_process(&process))
            );
            // (Microsecond, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000.into_process(&process))
            );
            // (Microsecond, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Microsecond, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Microsecond, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Microsecond, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Microsecond, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );

            // (Nanosecond, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(5.into_process(&process))
            );
            // (Nanosecond, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1.into_process(&process))
            );
            // (Nanosecond, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000.into_process(&process))
            );
            // (Nanosecond, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000.into_process(&process))
            );
            // (Nanosecond, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Nanosecond, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000.into_process(&process))
            );
            // (Nanosecond, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000.into_process(&process))
            );

            // (Native, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(5_000_000.into_process(&process))
            );
            // (Native, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000.into_process(&process))
            );
            // (Native, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Native, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Native, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Native, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Native, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );

            // (Native, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(5_000_000.into_process(&process))
            );
            // (Native, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000.into_process(&process))
            );
            // (Native, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Native, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Native, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Native, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Native, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    small_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
        });
    }
}

mod with_big_integer {
    use super::*;

    use num_traits::Num;

    #[test]
    fn without_valid_units_errors_badarg() {
        with_process(|process| {
            let big_integer_term: Term =
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process);

            assert_eq!(big_integer_term.tag(), Boxed);

            let valid_unit_term = Term::str_to_atom("native", DoNotCare).unwrap();
            let invalid_unit_term = Term::str_to_atom("s", DoNotCare).unwrap();

            assert_badarg!(erlang::convert_time_unit_3(
                big_integer_term,
                valid_unit_term,
                invalid_unit_term,
                &process,
            ));

            assert_badarg!(erlang::convert_time_unit_3(
                big_integer_term,
                invalid_unit_term,
                valid_unit_term,
                &process,
            ));
        });
    }

    #[test]
    fn with_valid_units_returns_converted_value() {
        with_process(|process| {
            let big_integer_term: Term =
                <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process);

            assert_eq!(big_integer_term.tag(), Boxed);

            // (Hertz, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    2_usize.into_process(&process),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("2_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Hertz, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Hertz, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Hertz, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Hertz, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Hertz, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Hertz, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    2_usize.into_process(&process),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );

            // (Second, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("5_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Second, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Second, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Second, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Second, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Second, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Second, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );

            // (Millisecond, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Millisecond, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Millisecond, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Millisecond, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Millisecond, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Millisecond, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Millisecond, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );

            // (Microsecond, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Microsecond, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Microsecond, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Microsecond, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Microsecond, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Microsecond, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Microsecond, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );

            // (Nanosecond, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("5_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Nanosecond, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(1_000_000_000.into_process(&process))
            );
            // (Nanosecond, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Nanosecond, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Nanosecond, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Nanosecond, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Nanosecond, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );

            // (Native, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Native, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Native, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Native, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Native, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Native, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Native, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );

            // (Native, Hertz)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    5_usize.into_process(&process),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Native, Second)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("second", DoNotCare).unwrap(),
                    &process
                ),
                Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process))
            );
            // (Native, Millisecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("millisecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Native, Microsecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("microsecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Native, Nanosecond)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("nanosecond", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Native, Native)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
            // (Native, PerformanceCounter)
            assert_eq!(
                erlang::convert_time_unit_3(
                    big_integer_term,
                    Term::str_to_atom("native", DoNotCare).unwrap(),
                    Term::str_to_atom("perf_counter", DoNotCare).unwrap(),
                    &process
                ),
                Ok(
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&process)
                )
            );
        });
    }
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|process| Term::slice_to_binary(&[1], &process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0, 1], &process);
        Term::subbinary(original, 1, 0, 1, 0, &process)
    });
}

fn errors_badarg<F>(time: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| {
        let from_unit = Term::str_to_atom("native", DoNotCare).unwrap();
        let to_unit = Term::str_to_atom("native", DoNotCare).unwrap();

        erlang::convert_time_unit_3(time(&process), from_unit, to_unit, &process)
    });
}

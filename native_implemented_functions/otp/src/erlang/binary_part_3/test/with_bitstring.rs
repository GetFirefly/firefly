use super::*;

use proptest::strategy::Strategy;

mod with_bit_count;
mod without_bit_count;

#[test]
fn without_integer_start_without_integer_length_errors_badarg() {
    run!(
        strategy::without_integer_start_without_integer_length,
        |(arc_process, binary, start, length)| {
            prop_assert_is_not_non_negative_integer!(
                result(&arc_process, binary, start, length),
                start
            );

            Ok(())
        },
    );
}

#[test]
fn without_integer_start_with_integer_length_errors_badarg() {
    run!(strategy::without_integer_start_with_integer_length, |(
        arc_process,
        binary,
        start,
        length,
    )| {
        prop_assert_is_not_non_negative_integer!(
            result(&arc_process, binary, start, length),
            start
        );

        Ok(())
    },);
}

#[test]
fn with_non_negative_integer_start_without_integer_length_errors_badarg() {
    run!(
        strategy::with_non_negative_integer_start_without_integer_length,
        |(arc_process, binary, start, length)| {
            prop_assert_is_not_integer!(result(&arc_process, binary, start, length), length);

            Ok(())
        },
    );
}

#[test]
fn with_negative_start_with_valid_length_errors_badarg() {
    run!(strategy::with_negative_start_with_valid_length, |(
        arc_process,
        binary,
        start,
        length,
    )| {
        prop_assert_is_not_non_negative_integer!(
            result(&arc_process, binary, start, length),
            start
        );

        Ok(())
    },);
}

#[test]
fn with_start_greater_than_size_with_non_negative_length_errors_badarg() {
    run!(
        strategy::with_start_greater_than_size_with_non_negative_length,
        |(arc_process, binary, start, length)| {
            prop_assert_badarg!(
                result(&arc_process, binary, start, length),
                format!("start ({}) exceeds available_byte_count", start)
            );

            Ok(())
        },
    );
}

#[test]
fn with_start_less_than_size_with_negative_length_past_start_errors_badarg() {
    run!(
        strategy::with_start_less_than_size_with_negative_length_past_start,
        |(arc_process, binary, start, length, end)| {
            prop_assert_badarg!(
                result(&arc_process, binary, start, length),
                format!("end ({}) is less than or equal to 0", end)
            );

            Ok(())
        },
    );
}

#[test]
fn with_start_less_than_size_with_positive_length_past_end_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_bitstring::with_byte_len_range(
                    strategy::NON_EMPTY_RANGE_INCLUSIVE.into(),
                    arc_process.clone(),
                ),
            )
                .prop_flat_map(|(arc_process, binary)| {
                    (
                        Just(arc_process.clone()),
                        Just(binary),
                        0..total_byte_len(binary),
                    )
                })
                .prop_map(|(arc_process, binary, start)| {
                    let length = total_byte_len(binary) - start + 1;
                    let end = start + length;

                    (
                        arc_process.clone(),
                        binary,
                        arc_process.integer(start).unwrap(),
                        arc_process.integer(length).unwrap(),
                        end,
                    )
                })
        },
        |(arc_process, binary, start, length, end)| {
            prop_assert_badarg!(
                result(&arc_process, binary, start, length),
                format!("end ({}) exceeds available_byte_count", end)
            );

            Ok(())
        },
    );
}

#[test]
fn with_positive_start_and_negative_length_returns_subbinary() {
    crate::test::with_positive_start_and_negative_length_returns_subbinary(
        file!(),
        returns_subbinary,
    );
}

fn returns_subbinary(
    (arc_process, binary, start, length): (Arc<Process>, Term, Term, Term),
) -> TestCaseResult {
    let result = result(&arc_process, binary, start, length);

    prop_assert!(result.is_ok());

    let returned = result.unwrap();

    prop_assert!(returned.is_boxed_subbinary());

    Ok(())
}

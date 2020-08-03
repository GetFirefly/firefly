use super::*;

#[test]
fn with_positive_start_and_positive_length_returns_subbinary() {
    crate::test::with_positive_start_and_positive_length_returns_subbinary(
        file!(),
        returns_subbinary,
    );
}

#[test]
fn with_size_start_and_negative_size_length_returns_binary() {
    crate::test::with_size_start_and_negative_size_length_returns_binary(file!(), returns_binary);
}

#[test]
fn with_zero_start_and_size_length_returns_binary() {
    crate::test::with_zero_start_and_size_length_returns_binary(file!(), returns_binary);
}

fn returns_binary(
    (arc_process, binary, start, length): (Arc<Process>, Term, Term, Term),
) -> TestCaseResult {
    let start_length = arc_process.tuple_from_slice(&[start, length]).unwrap();

    prop_assert_eq!(result(&arc_process, binary, start_length), Ok(binary));

    let returned_binary = result(&arc_process, binary, start_length).unwrap();

    prop_assert_eq!(
        returned_binary.is_boxed_subbinary(),
        binary.is_boxed_subbinary()
    );

    Ok(())
}

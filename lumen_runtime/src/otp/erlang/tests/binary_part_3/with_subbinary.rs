use super::*;

#[test]
fn without_integer_start_without_integer_length_errors_badarg() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 2, 1, &mut process);
        let start = Term::slice_to_tuple(
            &[0.into_process(&mut process), 0.into_process(&mut process)],
            &mut process,
        );
        let length = Term::str_to_atom("all", DoNotCare).unwrap();

        assert_badarg!(erlang::binary_part_3(binary, start, length, &mut process));
    });
}

#[test]
fn without_integer_start_with_integer_length_errors_badarg() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 2, 1, &mut process);
        let start = 0.into_process(&mut process);
        let length = Term::str_to_atom("all", DoNotCare).unwrap();

        assert_badarg!(erlang::binary_part_3(binary, start, length, &mut process));
    });
}

#[test]
fn with_integer_start_without_integer_length_errors_badarg() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 2, 1, &mut process);
        let start = 0.into_process(&mut process);
        let length = Term::str_to_atom("all", DoNotCare).unwrap();

        assert_badarg!(erlang::binary_part_3(binary, start, length, &mut process));
    });
}

#[test]
fn with_negative_start_with_valid_length_errors_badarg() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 2, 1, &mut process);
        let start = (-1isize).into_process(&mut process);
        let length = 0.into_process(&mut process);

        assert_badarg!(erlang::binary_part_3(binary, start, length, &mut process));
    });
}

#[test]
fn with_start_greater_than_size_with_non_negative_length_errors_badarg() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 0, 1, &mut process);
        let start = 1.into_process(&mut process);
        let length = 0.into_process(&mut process);

        assert_badarg!(erlang::binary_part_3(binary, start, length, &mut process));
    });
}

#[test]
fn with_start_less_than_size_with_negative_length_past_start_errors_badarg() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 2, 1, &mut process);
        let start = 0.into_process(&mut process);
        let length = (-1isize).into_process(&mut process);

        assert_badarg!(erlang::binary_part_3(binary, start, length, &mut process));
    });
}

#[test]
fn with_start_less_than_size_with_positive_length_past_end_errors_badarg() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 1, 1, &mut process);
        let start = 0.into_process(&mut process);
        let length = 2.into_process(&mut process);

        assert_badarg!(erlang::binary_part_3(binary, start, length, &mut process));
    });
}

#[test]
fn with_zero_start_and_byte_count_length_returns_new_subbinary_with_bytes() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 2, 1, &mut process);
        let start = 0.into_process(&mut process);
        let length = 2.into_process(&mut process);

        assert_eq!(
            erlang::binary_part_3(binary, start, length, &mut process),
            Ok(Term::subbinary(original, 0, 7, 2, 0, &mut process))
        );
    });
}

#[test]
fn with_byte_count_start_and_negative_byte_count_length_returns_new_subbinary_with_bytes() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary = Term::subbinary(original, 0, 7, 2, 1, &mut process);
        let start = 2.into_process(&mut process);
        let length = (-2isize).into_process(&mut process);

        assert_eq!(
            erlang::binary_part_3(binary, start, length, &mut process),
            Ok(Term::subbinary(original, 0, 7, 2, 0, &mut process))
        );
    });
}

#[test]
fn with_positive_start_and_negative_length_returns_subbinary() {
    with_process(|mut process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110], &mut process);
        let binary = Term::subbinary(original, 0, 7, 1, 0, &mut process);
        let start = 1.into_process(&mut process);
        let length = (-1isize).into_process(&mut process);

        assert_eq!(
            erlang::binary_part_3(binary, start, length, &mut process),
            Ok(Term::slice_to_binary(&[0b1111_1111], &mut process))
        );

        let returned_boxed = erlang::binary_part_3(binary, start, length, &mut process).unwrap();

        assert_eq!(returned_boxed.tag(), Boxed);

        let returned_unboxed: &Term = returned_boxed.unbox_reference();

        assert_eq!(returned_unboxed.tag(), Subbinary);
    });
}

#[test]
fn with_positive_start_and_positive_length_returns_subbinary() {
    with_process(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let binary: Term = process.subbinary(original, 0, 7, 2, 1).into();
        let start = 1.into_process(&mut process);
        let length = 1.into_process(&mut process);

        assert_eq!(
            erlang::binary_part_3(binary, start, length, &mut process),
            Ok(Term::slice_to_binary(&[0b0101_0101], &mut process))
        );

        let returned_boxed = erlang::binary_part_3(binary, start, length, &mut process).unwrap();

        assert_eq!(returned_boxed.tag(), Boxed);

        let returned_unboxed: &Term = returned_boxed.unbox_reference();

        assert_eq!(returned_unboxed.tag(), Subbinary);
    });
}

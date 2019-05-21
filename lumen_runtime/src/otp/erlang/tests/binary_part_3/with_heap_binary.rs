use super::*;

#[test]
fn without_integer_start_without_integer_length_errors_badarg() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[], &process);
        let start = Term::slice_to_tuple(
            &[0.into_process(&process), 0.into_process(&process)],
            &process,
        );
        let length = Term::str_to_atom("all", DoNotCare).unwrap();

        assert_badarg!(erlang::binary_part_3(binary, start, length, &process));
    });
}

#[test]
fn without_integer_start_with_integer_length_errors_badarg() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[], &process);
        let start = 0.into_process(&process);
        let length = Term::str_to_atom("all", DoNotCare).unwrap();

        assert_badarg!(erlang::binary_part_3(binary, start, length, &process));
    });
}

#[test]
fn with_integer_start_without_integer_length_errors_badarg() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[], &process);
        let start = 0.into_process(&process);
        let length = Term::str_to_atom("all", DoNotCare).unwrap();

        assert_badarg!(erlang::binary_part_3(binary, start, length, &process));
    });
}

#[test]
fn with_negative_start_with_valid_length_errors_badarg() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[], &process);
        let start = (-1isize).into_process(&process);
        let length = 0.into_process(&process);

        assert_badarg!(erlang::binary_part_3(binary, start, length, &process));
    });
}

#[test]
fn with_start_greater_than_size_with_non_negative_length_errors_badarg() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[], &process);
        let start = 1.into_process(&process);
        let length = 0.into_process(&process);

        assert_badarg!(erlang::binary_part_3(binary, start, length, &process));
    });
}

#[test]
fn with_start_less_than_size_with_negative_length_past_start_errors_badarg() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[0], &process);
        let start = 0.into_process(&process);
        let length = (-1isize).into_process(&process);

        assert_badarg!(erlang::binary_part_3(binary, start, length, &process));
    });
}

#[test]
fn with_start_less_than_size_with_positive_length_past_end_errors_badarg() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[0], &process);
        let start = 0.into_process(&process);
        let length = 2.into_process(&process);

        assert_badarg!(erlang::binary_part_3(binary, start, length, &process));
    });
}

#[test]
fn with_zero_start_and_size_length_returns_binary() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[0], &process);
        let start = 0.into_process(&process);
        let length = 1.into_process(&process);

        assert_eq!(
            erlang::binary_part_3(binary, start, length, &process),
            Ok(binary)
        );

        let returned_binary = erlang::binary_part_3(binary, start, length, &process).unwrap();

        assert_eq!(returned_binary.tagged, binary.tagged);
    });
}

#[test]
fn with_size_start_and_negative_size_length_returns_binary() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[0], &process);
        let start = 1.into_process(&process);
        let length = (-1isize).into_process(&process);

        assert_eq!(
            erlang::binary_part_3(binary, start, length, &process),
            Ok(binary)
        );

        let returned_binary = erlang::binary_part_3(binary, start, length, &process).unwrap();

        assert_eq!(returned_binary.tagged, binary.tagged);
    });
}

#[test]
fn with_positive_start_and_negative_length_returns_subbinary() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[0, 1], &process);
        let start = 1.into_process(&process);
        let length = (-1isize).into_process(&process);

        assert_eq!(
            erlang::binary_part_3(binary, start, length, &process),
            Ok(Term::slice_to_binary(&[0], &process))
        );

        let returned_boxed = erlang::binary_part_3(binary, start, length, &process).unwrap();

        assert_eq!(returned_boxed.tag(), Boxed);

        let returned_unboxed: &Term = returned_boxed.unbox_reference();

        assert_eq!(returned_unboxed.tag(), Subbinary);
    });
}

#[test]
fn with_positive_start_and_positive_length_returns_subbinary() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[0, 1], &process);
        let start = 1.into_process(&process);
        let length = 1.into_process(&process);

        assert_eq!(
            erlang::binary_part_3(binary, start, length, &process),
            Ok(Term::slice_to_binary(&[1], &process))
        );

        let returned_boxed = erlang::binary_part_3(binary, start, length, &process).unwrap();

        assert_eq!(returned_boxed.tag(), Boxed);

        let returned_unboxed: &Term = returned_boxed.unbox_reference();

        assert_eq!(returned_unboxed.tag(), Subbinary);
    });
}

use super::*;

#[test]
fn with_zero_returns_empty_prefix_and_subbinary() {
    with_process(|process| {
        let binary = bitstring!(1 :: 1, &process);
        let position = 0.into_process(&process);

        assert_eq!(
            erlang::split_binary_2(binary, position, &process),
            Ok(Term::slice_to_tuple(
                &[Term::slice_to_binary(&[], &process), binary],
                &process
            ))
        );
    });
}

#[test]
fn with_less_than_byte_len_returns_binary_prefix_and_suffix_bitstring() {
    with_process(|process| {
        let binary = bitstring!(1, 2 :: 2, &process);
        let position = 1.into_process(&process);

        assert_eq!(
            erlang::split_binary_2(binary, position, &process),
            Ok(Term::slice_to_tuple(
                &[
                    Term::slice_to_binary(&[1], &process),
                    bitstring!(2 :: 2, &process)
                ],
                &process
            ))
        )
    })
}

#[test]
fn with_byte_len_without_bit_count_returns_subbinary_and_empty_suffix() {
    with_process(|process| {
        let original = Term::slice_to_binary(&[1], &process);
        let binary = Term::subbinary(original, 0, 0, 1, 0, &process);
        let position = 1.into_process(&process);

        assert_eq!(
            erlang::split_binary_2(binary, position, &process),
            Ok(Term::slice_to_tuple(
                &[binary, Term::slice_to_binary(&[], &process)],
                &process
            ))
        );
    });
}

#[test]
fn with_byte_len_with_bit_count_errors_badarg() {
    with_process(|process| {
        let binary = bitstring!(1, 2 :: 2, &process);
        let position = 2.into_process(&process);

        assert_badarg!(erlang::split_binary_2(binary, position, &process));
    });
}

#[test]
fn with_greater_than_byte_len_errors_badarg() {
    with_process(|process| {
        let binary = bitstring!(1, 2 :: 2, &process);
        let position = 3.into_process(&process);

        assert_badarg!(erlang::split_binary_2(binary, position, &process));
    });
}

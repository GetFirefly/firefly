use super::*;

#[test]
fn with_zero_returns_empty_prefix_and_binary() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[1], &process);
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
fn with_less_than_byte_len_returns_binary_prefix_and_suffix_binary() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[1, 2], &process);
        let position = 1.into_process(&process);

        assert_eq!(
            erlang::split_binary_2(binary, position, &process),
            Ok(Term::slice_to_tuple(
                &[
                    Term::slice_to_binary(&[1], &process),
                    Term::slice_to_binary(&[2], &process)
                ],
                &process
            ))
        )
    })
}

#[test]
fn with_byte_len_returns_subbinary_and_empty_suffix() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[1], &process);
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
fn with_greater_than_byte_len_errors_badarg() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[1], &process);
        let position = 2.into_process(&process);

        assert_badarg!(erlang::split_binary_2(binary, position, &process));
    });
}

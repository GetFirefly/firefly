use super::*;

#[test]
fn with_less_than_byte_len_returns_binary_prefix_and_suffix_bitstring() {
    with_process(|process| {
        let binary = bitstring!(1, 2 :: 2, &process);
        let position = process.integer(1).unwrap();

        assert_eq!(
            erlang::split_binary_2(binary, position, &process),
            Ok(process
                .tuple_from_slice(&[
                    process.binary_from_bytes(&[1]).unwrap(),
                    bitstring!(2 :: 2, &process)
                ],)
                .unwrap())
        )
    })
}

#[test]
fn with_byte_len_without_bit_count_returns_subbinary_and_empty_suffix() {
    with_process(|process| {
        let original = process.binary_from_bytes(&[1]).unwrap();
        let binary = process
            .subbinary_from_original(original, 0, 0, 1, 0)
            .unwrap();
        let position = process.integer(1).unwrap();

        assert_eq!(
            erlang::split_binary_2(binary, position, &process),
            Ok(process
                .tuple_from_slice(&[binary, process.binary_from_bytes(&[]).unwrap()],)
                .unwrap())
        );
    });
}

#[test]
fn with_byte_len_with_bit_count_errors_badarg() {
    with_process(|process| {
        let binary = bitstring!(1, 2 :: 2, &process);
        let position = process.integer(2).unwrap();

        assert_badarg!(erlang::split_binary_2(binary, position, &process));
    });
}

#[test]
fn with_greater_than_byte_len_errors_badarg() {
    with_process(|process| {
        let binary = bitstring!(1, 2 :: 2, &process);
        let position = process.integer(3).unwrap();

        assert_badarg!(erlang::split_binary_2(binary, position, &process));
    });
}

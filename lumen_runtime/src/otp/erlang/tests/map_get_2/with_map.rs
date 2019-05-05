use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_key() {
    with_process(|process| {
        let key = Term::str_to_atom("key", DoNotCare).unwrap();
        let value = Term::str_to_atom("value", DoNotCare).unwrap();
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = Term::str_to_atom("non_key", DoNotCare).unwrap();

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_local_reference_key() {
    with_process(|process| {
        let key = Term::next_local_reference(process);
        let value = Term::next_local_reference(process);
        let map_with_key = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map_with_key, &process), Ok(value));

        let map_without_key = Term::slice_to_map(&[], &process);

        assert_badkey!(
            erlang::map_get_2(key, map_without_key, &process),
            key,
            &process
        );
    });
}

#[test]
fn with_empty_list_key() {
    with_process(|process| {
        let key = Term::EMPTY_LIST;
        let value = Term::EMPTY_LIST;
        let map_with_key = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map_with_key, &process), Ok(value));

        let map_without_key = Term::slice_to_map(&[], &process);

        assert_badkey!(
            erlang::map_get_2(key, map_without_key, &process),
            key,
            &process
        );
    });
}

#[test]
fn with_list_key() {
    with_process(|process| {
        let key = Term::cons(0.into_process(&process), Term::EMPTY_LIST, &process);
        let value = Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process);
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = Term::cons(2.into_process(&process), Term::EMPTY_LIST, &process);

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_small_key_integer() {
    with_process(|process| {
        let key = 0.into_process(&process);
        let value = 1.into_process(&process);
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = 2.into_process(&process);

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_big_key_integer() {
    with_process(|process| {
        let key = <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process);
        let value = <BigInt as Num>::from_str_radix("576460752303423490", 10)
            .unwrap()
            .into_process(&process);
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = <BigInt as Num>::from_str_radix("576460752303423491", 10)
            .unwrap()
            .into_process(&process);

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_float_key() {
    with_process(|process| {
        let key = 1.0.into_process(&process);
        let value = 2.0.into_process(&process);
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = 3.0.into_process(&process);

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_local_key_pid() {
    with_process(|process| {
        let key = Term::local_pid(0, 1).unwrap();
        let value = Term::local_pid(2, 3).unwrap();
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = Term::local_pid(4, 5).unwrap();

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_external_key_pid() {
    with_process(|process| {
        let key = Term::external_pid(1, 2, 3, &process).unwrap();
        let value = Term::external_pid(4, 5, 6, &process).unwrap();
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = Term::external_pid(7, 8, 9, &process).unwrap();

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_tuple_key() {
    with_process(|process| {
        let key = Term::slice_to_tuple(&[0.into_process(&process)], &process);
        let value = Term::slice_to_tuple(&[1.into_process(&process)], &process);
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = Term::slice_to_tuple(&[2.into_process(&process)], &process);

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_heap_key_binary() {
    with_process(|process| {
        let key = Term::slice_to_binary(&[0], &process);
        let value = Term::slice_to_binary(&[1], &process);
        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        let non_key = Term::slice_to_binary(&[2], &process);

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

#[test]
fn with_subbinary_key() {
    with_process(|process| {
        // <<1::1, 2>>
        let key_original = Term::slice_to_binary(&[129, 0b0000_0000], &process);
        let key = Term::subbinary(key_original, 0, 1, 1, 0, &process);

        // <<3::3, 4>>
        let value_original = Term::slice_to_binary(&[96, 0b0000_0000], &process);
        let value = Term::subbinary(value_original, 0, 3, 1, 0, &process);

        let map = Term::slice_to_map(&[(key, value)], &process);

        assert_eq!(erlang::map_get_2(key, map, &process), Ok(value));

        // <<5::5, 6>>
        let non_key_original = Term::slice_to_binary(&[40, 0b00110_000], &process);
        let non_key = Term::subbinary(non_key_original, 0, 5, 1, 0, &process);

        assert_badkey!(erlang::map_get_2(non_key, map, &process), non_key, &process);
    });
}

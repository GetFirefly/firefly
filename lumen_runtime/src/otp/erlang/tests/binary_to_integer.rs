use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_returns_bad_argument() {
    let mut process: Process = Default::default();
    let atom_term = Term::str_to_atom("😈🤘", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_integer(atom_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_empty_list_returns_bad_argument() {
    let mut process: Process = Default::default();

    assert_eq_in_process!(
        erlang::binary_to_integer(Term::EMPTY_LIST, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let list_term = list_term(&mut process);

    assert_eq_in_process!(
        erlang::binary_to_integer(list_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let small_integer_term = 0usize.into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_to_integer(small_integer_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let big_integer_term: Term = <BigInt as Num>::from_str_radix("18446744073709551616", 10)
        .unwrap()
        .into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_to_integer(big_integer_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_to_integer(float_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_integer(tuple_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_heap_binary_with_min_small_integer_returns_small_integer() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary("-576460752303423488".as_bytes(), &mut process);

    let integer_result = erlang::binary_to_integer(heap_binary_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );
    assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
}

#[test]
fn with_heap_binary_with_max_small_integer_returns_small_integer() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary("576460752303423487".as_bytes(), &mut process);

    let integer_result = erlang::binary_to_integer(heap_binary_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );
    assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
}

#[test]
fn with_heap_binary_with_less_than_min_small_integer_returns_big_integer() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary("-576460752303423489".as_bytes(), &mut process);

    let integer_result = erlang::binary_to_integer(heap_binary_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Tag::Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), Tag::BigInteger);
}

#[test]
fn with_heap_binary_with_greater_than_max_small_integer_returns_big_integer() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary("576460752303423488".as_bytes(), &mut process);

    let integer_result = erlang::binary_to_integer(heap_binary_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Tag::Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), Tag::BigInteger);
}

#[test]
fn with_heap_binary_with_non_decimal_returns_bad_argument() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary("FF".as_bytes(), &mut process);

    assert_eq_in_process!(
        erlang::binary_to_integer(heap_binary_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_subbinary_with_min_small_integer_returns_small_integer() {
    let mut process: Process = Default::default();
    // <<1::1, "-576460752303423488">>
    let heap_binary_term = Term::slice_to_binary(
        &[
            150,
            154,
            155,
            155,
            26,
            27,
            24,
            27,
            154,
            153,
            25,
            152,
            25,
            154,
            25,
            25,
            154,
            28,
            28,
            0b0000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 19, 0, &mut process);

    let integer_result = erlang::binary_to_integer(subbinary_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );
    assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
}

#[test]
fn with_subbinary_with_max_small_integer_returns_small_integer() {
    let mut process: Process = Default::default();
    // <<1::1, "576460752303423487">>
    let heap_binary_term = Term::slice_to_binary(
        &[
            154,
            155,
            155,
            26,
            27,
            24,
            27,
            154,
            153,
            25,
            152,
            25,
            154,
            25,
            25,
            154,
            28,
            27,
            0b1000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 18, 0, &mut process);

    let integer_result = erlang::binary_to_integer(subbinary_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );
    assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
}

#[test]
fn with_subbinary_with_less_than_min_small_integer_returns_big_integer() {
    let mut process: Process = Default::default();
    // <<1::1, "-576460752303423489">>
    let heap_binary_term = Term::slice_to_binary(
        &[
            150,
            154,
            155,
            155,
            26,
            27,
            24,
            27,
            154,
            153,
            25,
            152,
            25,
            154,
            25,
            25,
            154,
            28,
            28,
            0b1000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 19, 0, &mut process);

    let integer_result = erlang::binary_to_integer(subbinary_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Tag::Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), Tag::BigInteger);
}

#[test]
fn with_subbinary_with_greater_than_max_small_integer_returns_big_integer() {
    let mut process: Process = Default::default();
    // <<1::1, "576460752303423488">>
    let heap_binary_term = Term::slice_to_binary(
        &[
            154,
            155,
            155,
            26,
            27,
            24,
            27,
            154,
            153,
            25,
            152,
            25,
            154,
            25,
            25,
            154,
            28,
            28,
            0b0000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 18, 0, &mut process);

    let integer_result = erlang::binary_to_integer(subbinary_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Tag::Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), Tag::BigInteger);
}

#[test]
fn with_subbinary_with_non_decimal_returns_bad_argument() {
    let mut process: Process = Default::default();
    // <<1:1, "FF>>
    let heap_binary_term = Term::slice_to_binary(&[163, 35, 0b000_0000], &mut process);
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 2, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_integer(subbinary_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

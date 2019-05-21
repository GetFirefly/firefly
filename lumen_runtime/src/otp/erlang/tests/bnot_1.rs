use super::*;

use num_traits::Num;

#[test]
fn with_atom_errors_badarith() {
    errors_badarith(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarith() {
    errors_badarith(|process| Term::next_local_reference(process));
}

#[test]
fn with_heap_binary_errors_badarith() {
    errors_badarith(|process| Term::slice_to_binary(&[0], &process));
}

#[test]
fn with_subbinary_errors_badarith() {
    errors_badarith(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

#[test]
fn with_empty_list_errors_badarith() {
    errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarith() {
    errors_badarith(|process| list_term(&process));
}

#[test]
fn with_small_integer_returns_small_integer() {
    with_process(|process| {
        let integer = 0b10.into_process(&process);

        assert_eq!(
            erlang::bnot_1(integer, &process),
            Ok((-3).into_process(&process))
        );
    });
}

#[test]
fn with_big_integer_returns_big_integer() {
    with_process(|process| {
        let integer = <BigInt as Num>::from_str_radix(
            "1010101010101010101010101010101010101010101010101010101010101010",
            2,
        )
        .unwrap()
        .into_process(&process);

        assert_eq!(integer.tag(), Boxed);

        let unboxed_integer: &Term = integer.unbox_reference();

        assert_eq!(unboxed_integer.tag(), BigInteger);

        assert_eq!(
            erlang::bnot_1(integer, &process),
            Ok(
                <BigInt as Num>::from_str_radix("-12297829382473034411", 10,)
                    .unwrap()
                    .into_process(&process)
            )
        );
    });
}

#[test]
fn with_float_that_is_negative_returns_positive() {
    errors_badarith(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarith() {
    errors_badarith(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarith() {
    errors_badarith(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarith() {
    errors_badarith(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarith() {
    errors_badarith(|process| Term::slice_to_map(&[], &process));
}

fn errors_badarith<I>(integer: I)
where
    I: FnOnce(&Process) -> Term,
{
    super::errors_badarith(|process| erlang::bnot_1(integer(&process), &process));
}

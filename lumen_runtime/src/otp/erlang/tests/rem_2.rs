use super::*;

mod with_big_integer_dividend;
mod with_float_dividend;
mod with_small_integer_dividend;

#[test]
fn with_atom_dividend_errors_badarith() {
    with_dividend_errors_badarith(|_| Term::str_to_atom("dividend", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_dividend_errors_badarith() {
    with_dividend_errors_badarith(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_dividend_errors_badarith() {
    with_dividend_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_dividend_errors_badarith() {
    with_dividend_errors_badarith(|mut process| {
        Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_local_pid_dividend_errors_badarith() {
    with_dividend_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_dividend_errors_badarith() {
    with_dividend_errors_badarith(|mut process| Term::external_pid(1, 2, 3, &mut process).unwrap());
}

#[test]
fn with_tuple_dividend_errors_badarith() {
    with_dividend_errors_badarith(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_dividend_errors_badarith() {
    with_dividend_errors_badarith(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_dividend_errors_badarith() {
    with_dividend_errors_badarith(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_dividend_errors_badarith() {
    with_dividend_errors_badarith(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

fn with_dividend_errors_badarith<M>(dividend: M)
where
    M: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| {
        let dividend = dividend(&mut process);
        let divisor = 0.into_process(&mut process);

        erlang::rem_2(dividend, divisor, &mut process)
    });
}

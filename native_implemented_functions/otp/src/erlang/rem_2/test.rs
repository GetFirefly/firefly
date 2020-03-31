mod with_big_integer_dividend;
mod with_small_integer_dividend;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Just};

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::rem_2::native;
use crate::runtime::scheduler::SchedulerDependentAlloc;
use crate::test::with_process;
use crate::test::{external_arc_node, strategy};

#[test]
fn without_integer_dividend_errors_badarith() {
    crate::test::without_integer_dividend_errors_badarith(file!(), native);
}

#[test]
fn with_integer_dividend_without_integer_divisor_errors_badarith() {
    crate::test::with_integer_dividend_without_integer_divisor_errors_badarith(file!(), native);
}

#[test]
fn with_integer_dividend_with_zero_divisor_errors_badarith() {
    crate::test::with_integer_dividend_with_zero_divisor_errors_badarith(file!(), native);
}

#[test]
fn with_atom_dividend_errors_badarith() {
    with_dividend_errors_badarith(|_| Atom::str_to_term("dividend"));
}

#[test]
fn with_local_reference_dividend_errors_badarith() {
    with_dividend_errors_badarith(|process| process.next_reference().unwrap());
}

#[test]
fn with_empty_list_dividend_errors_badarith() {
    with_dividend_errors_badarith(|_| Term::NIL);
}

#[test]
fn with_list_dividend_errors_badarith() {
    with_dividend_errors_badarith(|process| {
        process
            .cons(process.integer(0).unwrap(), process.integer(1).unwrap())
            .unwrap()
    });
}

#[test]
fn with_local_pid_dividend_errors_badarith() {
    with_dividend_errors_badarith(|_| Pid::make_term(0, 1).unwrap());
}

#[test]
fn with_external_pid_dividend_errors_badarith() {
    with_dividend_errors_badarith(|process| {
        process.external_pid(external_arc_node(), 2, 3).unwrap()
    });
}

#[test]
fn with_tuple_dividend_errors_badarith() {
    with_dividend_errors_badarith(|process| process.tuple_from_slice(&[]).unwrap());
}

#[test]
fn with_map_is_dividend_errors_badarith() {
    with_dividend_errors_badarith(|process| process.map_from_slice(&[]).unwrap());
}

#[test]
fn with_heap_binary_dividend_errors_badarith() {
    with_dividend_errors_badarith(|process| process.binary_from_bytes(&[]).unwrap());
}

#[test]
fn with_subbinary_dividend_errors_badarith() {
    with_dividend_errors_badarith(|process| {
        let original = process
            .binary_from_bytes(&[0b0000_00001, 0b1111_1110, 0b1010_1011])
            .unwrap();
        process
            .subbinary_from_original(original, 0, 7, 2, 1)
            .unwrap()
    });
}

fn with_dividend_errors_badarith<M>(dividend: M)
where
    M: FnOnce(&Process) -> Term,
{
    errors_badarith(|process| {
        let dividend = dividend(&process);
        let divisor = process.integer(0).unwrap();

        native(&process, dividend, divisor)
    });
}

fn errors_badarith<F>(actual: F)
where
    F: FnOnce(&Process) -> exception::Result<Term>,
{
    with_process(|process| assert_badarith!(actual(&process)))
}

use super::*;

mod with_big_integer_dividend;
mod with_small_integer_dividend;

#[test]
fn without_integer_dividend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::rem_2(dividend, divisor, &arc_process),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_dividend_without_integer_divisor_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::rem_2(dividend, divisor, &arc_process),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_dividend_with_zero_divisor_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    Just(arc_process.integer(0).unwrap()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::rem_2(dividend, divisor, &arc_process),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_atom_dividend_errors_badarith() {
    with_dividend_errors_badarith(|_| atom_unchecked("dividend"));
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
    with_dividend_errors_badarith(|_| make_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_dividend_errors_badarith() {
    with_dividend_errors_badarith(|process| process.external_pid_with_node_id(1, 2, 3).unwrap());
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
    M: FnOnce(&ProcessControlBlock) -> Term,
{
    super::errors_badarith(|process| {
        let dividend = dividend(&process);
        let divisor = process.integer(0).unwrap();

        erlang::rem_2(dividend, divisor, &process)
    });
}

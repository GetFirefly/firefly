use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

mod with_list;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::list_to_binary_1(list, &arc_process), Err(badarg!()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|process| {
        assert_eq!(
            erlang::list_to_binary_1(Term::EMPTY_LIST, &process),
            Ok(Term::slice_to_binary(&[], &process))
        );
    });
}

// > Bin1 = <<1,2,3>>.
// <<1,2,3>>
// > Bin2 = <<4,5>>.
// <<4,5>>
// > Bin3 = <<6>>.
// <<6>>
// > list_to_binary([Bin1,1,[2,3,Bin2],4|Bin3]).
// <<1,2,3,1,2,3,4,5,4,6>>
#[test]
fn otp_doctest_returns_binary() {
    with_process(|process| {
        let bin1 = Term::slice_to_binary(&[1, 2, 3], &process);
        let bin2 = Term::slice_to_binary(&[4, 5], &process);
        let bin3 = Term::slice_to_binary(&[6], &process);

        let iolist = Term::slice_to_improper_list(
            &[
                bin1,
                1.into_process(&process),
                Term::slice_to_list(
                    &[2.into_process(&process), 3.into_process(&process), bin2],
                    &process,
                ),
                4.into_process(&process),
            ],
            bin3,
            &process,
        );

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(
                &[1, 2, 3, 1, 2, 3, 4, 5, 4, 6],
                &process
            ))
        )
    });
}

#[test]
fn with_recursive_lists_ofbinaries_and_bytes_ending_in_binary_or_empty_list_returns_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&top(arc_process.clone()), |list| {
                let result = erlang::list_to_binary_1(list, &arc_process);

                prop_assert!(result.is_ok());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| {
        let binary_term =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(binary_term, 0, 7, 2, 1, &process)
    });
}

fn byte(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    any::<u8>().prop_map(move |byte| byte.into_process(&arc_process))
}

fn container(
    element: impl Strategy<Value = Term>,
    arc_process: Arc<Process>,
) -> impl Strategy<Value = Term> {
    (
        proptest::collection::vec(element, 0..=3),
        tail(arc_process.clone()),
    )
        .prop_map(move |(element_vec, tail)| {
            Term::slice_to_improper_list(&element_vec, tail, &arc_process)
        })
}

fn errors_badarg<I>(iolist: I)
where
    I: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::list_to_binary_1(iolist(&process), &process))
}

fn leaf(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    prop_oneof![
        strategy::term::is_binary(arc_process.clone()),
        byte(arc_process),
    ]
}

fn recursive(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    leaf(arc_process.clone()).prop_recursive(3, 3 * 4, 3, move |element| {
        container(element, arc_process.clone())
    })
}

fn tail(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    prop_oneof![
        strategy::term::is_binary(arc_process),
        Just(Term::EMPTY_LIST)
    ]
}

fn top(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    strategy::term::list::intermediate(recursive(arc_process.clone()), (0..=3).into(), arc_process)
}

use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

mod with_list;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(
                    erlang::list_to_binary_1(list, &arc_process),
                    Err(badarg!().into())
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|process| {
        assert_eq!(
            erlang::list_to_binary_1(Term::NIL, &process),
            Ok(process.binary_from_bytes(&[]).unwrap())
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
        let bin1 = process.binary_from_bytes(&[1, 2, 3]).unwrap();
        let bin2 = process.binary_from_bytes(&[4, 5]).unwrap();
        let bin3 = process.binary_from_bytes(&[6]).unwrap();

        let iolist = process
            .improper_list_from_slice(
                &[
                    bin1,
                    process.integer(1).unwrap(),
                    process
                        .list_from_slice(&[
                            process.integer(2).unwrap(),
                            process.integer(3).unwrap(),
                            bin2,
                        ])
                        .unwrap(),
                    process.integer(4).unwrap(),
                ],
                bin3,
            )
            .unwrap();

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(process
                .binary_from_bytes(&[1, 2, 3, 1, 2, 3, 4, 5, 4, 6],)
                .unwrap())
        )
    });
}

#[test]
fn with_recursive_lists_of_binaries_and_bytes_ending_in_binary_or_empty_list_returns_binary() {
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

fn byte(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    any::<u8>()
        .prop_map(move |byte| arc_process.integer(byte).unwrap())
        .boxed()
}

fn container(element: BoxedStrategy<Term>, arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (
        proptest::collection::vec(element, 0..=3),
        tail(arc_process.clone()),
    )
        .prop_map(move |(element_vec, tail)| {
            arc_process
                .improper_list_from_slice(&element_vec, tail)
                .unwrap()
        })
        .boxed()
}

fn leaf(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::is_binary(arc_process.clone()),
        byte(arc_process),
    ]
    .boxed()
}

fn recursive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    leaf(arc_process.clone())
        .prop_recursive(3, 3 * 4, 3, move |element| {
            container(element, arc_process.clone())
        })
        .boxed()
}

fn tail(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![strategy::term::is_binary(arc_process), Just(Term::NIL)].boxed()
}

fn top(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (
        proptest::collection::vec(recursive(arc_process.clone()), 1..=4),
        tail(arc_process.clone()),
    )
        .prop_map(move |(element_vec, tail)| {
            arc_process
                .improper_list_from_slice(&element_vec, tail)
                .unwrap()
        })
        .boxed()
}

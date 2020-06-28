use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::{prop_assert, prop_oneof};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::iolist_size_1::result;
use crate::test::strategy::term::*;
use crate::test::strategy::*;
use crate::test::with_process;

#[test]
fn without_list_or_bitstring_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                is_not_list_or_bitstring(arc_process),
            )
        },
        |(arc_process, iolist_or_binary)| {
            prop_assert_is_not_type!(
                result(&arc_process, iolist_or_binary),
                iolist_or_binary,
                "an iolist (a maybe improper list with byte, binary, or iolist elements and binary or empty list tail) or binary"
            );

            Ok(())
        }
    );
}

#[test]
fn with_iolist_or_binary_returns_non_negative_integer() {
    run!(
        |arc_process| { (Just(arc_process.clone()), is_iolist_or_binary(arc_process)) },
        |(arc_process, iolist_or_binary)| {
            let result = result(&arc_process, iolist_or_binary);

            prop_assert!(result.is_ok(), "{:?}", result);

            let size = result.unwrap();

            prop_assert!(size.is_integer());
            prop_assert!(size >= arc_process.integer(0).unwrap());

            Ok(())
        }
    )
}

// > iolist_size([1,2|<<3,4>>]).
// 4
#[test]
fn otp_doctest() {
    with_process(|process| {
        let iolist = process
            .improper_list_from_slice(
                &[process.integer(1).unwrap(), process.integer(2).unwrap()],
                process.binary_from_bytes(&[3, 4]).unwrap(),
            )
            .unwrap();

        assert_eq!(result(process, iolist), Ok(process.integer(4).unwrap()))
    });
}

// > Bin1 = <<1,2,3>>.
// <<1,2,3>>
// > Bin2 = <<4,5>>.
// <<4,5>>
// > Bin3 = <<6>>.
// <<6>>
// > iolist_size([Bin1,1,[2,3,Bin2],4|Bin3]).
// 10
#[test]
fn with_iolist_returns_size() {
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

        assert_eq!(result(process, iolist), Ok(process.integer(10).unwrap()))
    });
}

#[test]
fn with_binary_returns_size() {
    with_process(|process| {
        let bin = process.binary_from_bytes(&[1, 2, 3]).unwrap();

        assert_eq!(result(process, bin), Ok(process.integer(3).unwrap()))
    });
}

#[test]
fn with_subbinary_in_list_returns_size() {
    with_process(|process| {
        let iolist = process
            .list_from_slice(&[process
                .subbinary_from_original(
                    process.binary_from_bytes(&[1, 2, 3, 4, 5]).unwrap(),
                    1,
                    0,
                    3,
                    0,
                )
                .unwrap()])
            .unwrap();

        assert_eq!(result(process, iolist), Ok(process.integer(3).unwrap()))
    });
}

#[test]
fn with_subbinary_returns_size() {
    with_process(|process| {
        let iolist = process
            .subbinary_from_original(
                process.binary_from_bytes(&[1, 2, 3, 4, 5]).unwrap(),
                1,
                0,
                3,
                0,
            )
            .unwrap();

        assert_eq!(result(process, iolist), Ok(process.integer(3).unwrap()))
    });
}

#[test]
fn with_procbin_returns_size() {
    with_process(|process| {
        let bytes = [7; 65];
        let procbin = process.binary_from_bytes(&bytes).unwrap();
        // We expect this to be a procbin, since it's > 64 bytes. Make sure it is.
        assert!(procbin.is_boxed_procbin());
        let iolist = process.list_from_slice(&[procbin]).unwrap();

        assert_eq!(result(process, iolist), Ok(process.integer(65).unwrap()))
    });
}

#[test]
fn with_improper_list_smallint_tail_errors_badarg() {
    with_process(|process| {
        let tail = process.integer(42).unwrap();
        let iolist = process
            .improper_list_from_slice(&[process.binary_from_bytes(&[1, 2, 3]).unwrap()], tail)
            .unwrap();

        assert_badarg!(
            result(process, iolist),
            format!(
                "iolist_or_binary ({}) tail ({}) cannot be a byte",
                iolist, tail
            )
        );
    });
}

pub fn is_not_list_or_bitstring(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let element = term(arc_process.clone());
    let size_range = size_range();

    prop_oneof![
        integer::big(arc_process.clone()),
        local_reference(arc_process.clone()),
        is_function(arc_process.clone()),
        float(arc_process.clone()),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        pid::local(),
        // TODO `LocalPort`,
        term::atom(),
        integer::small(arc_process.clone()),
        prop_oneof![
            tuple::intermediate(element.clone(), size_range.clone(), arc_process.clone()),
            map::intermediate(element.clone(), size_range, arc_process.clone()),
        ]
    ]
    .boxed()
}

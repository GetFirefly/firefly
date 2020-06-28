use std::convert::TryInto;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::iolist_to_iovec_1::result;
use crate::test::strategy::term::is_iolist_or_binary;
use crate::test::with_process;

#[test]
fn with_iolist_or_binary_returns_iovec() {
    run!(
        |arc_process| { (Just(arc_process.clone()), is_iolist_or_binary(arc_process)) },
        |(arc_process, iolist_or_binary)| {
            let result = result(&arc_process, iolist_or_binary);

            prop_assert!(
                result.is_ok(),
                "{} cannot be converted to a binary",
                iolist_or_binary
            );

            let iovec = result.unwrap();

            prop_assert!(iovec.is_list());

            let iovec_cons: Boxed<Cons> = iovec.try_into().unwrap();

            for iovec_result in iovec_cons.into_iter() {
                prop_assert!(iovec_result.is_ok());

                let iovec_element = iovec_result.unwrap();

                prop_assert!(iovec_element.is_binary());
            }

            Ok(())
        }
    )
}

// See https://github.com/erlang/otp/blob/cf6cf5e5f82e348ecb9bb02d70027fc4961aee3d/erts/emulator/test/iovec_SUITE.erl#L106-L115
#[test]
fn is_idempotent() {
    run!(
        |arc_process| { (Just(arc_process.clone()), is_iolist_or_binary(arc_process)) },
        |(arc_process, iolist_or_binary)| {
            let first_result = result(&arc_process, iolist_or_binary);

            prop_assert!(
                first_result.is_ok(),
                "{} cannot be converted to a binary",
                iolist_or_binary
            );

            let first = first_result.unwrap();

            prop_assert_eq!(result(&arc_process, first), Ok(first));

            Ok(())
        }
    )
}

#[test]
fn with_binary_returns_binary_in_list() {
    with_process(|process| {
        let bin = process.binary_from_bytes(&[1, 2, 3]).unwrap();

        assert_eq!(
            result(process, bin),
            Ok(process
                .list_from_slice(&[process.binary_from_bytes(&[1, 2, 3]).unwrap()])
                .unwrap())
        )
    });
}

#[test]
fn with_procbin_in_list_returns_list() {
    with_process(|process| {
        let bytes = [7; 65];
        let procbin = process.binary_from_bytes(&bytes).unwrap();
        // We expect this to be a procbin, since it's > 64 bytes. Make sure it is.
        assert!(procbin.is_boxed_procbin());
        let iolist = process.list_from_slice(&[procbin]).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process
                .list_from_slice(&[process.binary_from_bytes(&bytes).unwrap()])
                .unwrap())
        )
    });
}

#[test]
fn with_subbinary_in_list_returns_list() {
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

        assert_eq!(
            result(process, iolist),
            Ok(process
                .list_from_slice(&[process.binary_from_bytes(&[2, 3, 4]).unwrap()])
                .unwrap())
        )
    });
}

#[test]
fn with_subbinary_returns_list() {
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

        assert_eq!(
            result(process, iolist),
            Ok(process
                .list_from_slice(&[process.binary_from_bytes(&[2, 3, 4]).unwrap()])
                .unwrap())
        )
    });
}

#[test]
fn with_improper_list_smallint_tail_errors_badarg() {
    with_process(|process| {
        let tail = process.integer(42).unwrap();
        let iolist_or_binary = process
            .improper_list_from_slice(&[process.binary_from_bytes(&[1, 2, 3]).unwrap()], tail)
            .unwrap();

        assert_badarg!(
            result(process, iolist_or_binary),
            format!(
                "iolist_or_binary ({}) tail ({}) cannot be a byte",
                iolist_or_binary, tail
            )
        )
    });
}

// List elements must be smallint, binary, or lists thereof
#[test]
fn with_atom_in_iolist_errors_badarg() {
    with_process(|process| {
        let element = Atom::str_to_term("foo");
        let iolist_or_binary = process.list_from_slice(&[element]).unwrap();

        assert_badarg!(
            result(process, iolist_or_binary),
            format!(
                "iolist_or_binary ({}) element ({}) is not a byte, binary, or nested iolist",
                iolist_or_binary, element
            )
        )
    });
}

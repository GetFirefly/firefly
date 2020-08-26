use proptest::prop_assert;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::iolist_to_binary_1::result;
use crate::test::strategy::term::is_iolist_or_binary;
use crate::test::with_process;

#[test]
fn with_iolist_or_binary_returns_binary() {
    run!(
        |arc_process| { (Just(arc_process.clone()), is_iolist_or_binary(arc_process)) },
        |(arc_process, iolist_or_binary)| {
            let result = result(&arc_process, iolist_or_binary);

            prop_assert!(result.is_ok());

            let binary = result.unwrap();

            prop_assert!(binary.is_binary());

            Ok(())
        }
    )
}

// > Bin1 = <<1,2,3>>.
// <<1,2,3>>
// > Bin2 = <<4,5>>.
// <<4,5>>
// > Bin3 = <<6>>.
// <<6>>
// > iolist_to_binary([Bin1,1,[2,3,Bin2],4|Bin3]).
// <<1,2,3,1,2,3,4,5,4,6>>
#[test]
fn otp_doctest_returns_binary() {
    with_process(|process| {
        let bin1 = process.binary_from_bytes(&[1, 2, 3]);
        let bin2 = process.binary_from_bytes(&[4, 5]);
        let bin3 = process.binary_from_bytes(&[6]);

        let iolist = process.improper_list_from_slice(
            &[
                bin1,
                process.integer(1),
                process.list_from_slice(&[process.integer(2), process.integer(3), bin2]),
                process.integer(4),
            ],
            bin3,
        );

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[1, 2, 3, 1, 2, 3, 4, 5, 4, 6],))
        )
    });
}

#[test]
fn with_binary_returns_binary() {
    with_process(|process| {
        let bin = process.binary_from_bytes(&[1, 2, 3]);

        assert_eq!(
            result(process, bin),
            Ok(process.binary_from_bytes(&[1, 2, 3],))
        )
    });
}

#[test]
fn with_procbin_in_list_returns_binary() {
    with_process(|process| {
        let bytes = [7; 65];
        let procbin = process.binary_from_bytes(&bytes);
        // We expect this to be a procbin, since it's > 64 bytes. Make sure it is.
        assert!(procbin.is_boxed_procbin());
        let iolist = process.list_from_slice(&[procbin]);

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&bytes))
        )
    });
}

#[test]
fn with_subbinary_in_list_returns_binary() {
    with_process(|process| {
        let iolist = process.list_from_slice(&[process.subbinary_from_original(
            process.binary_from_bytes(&[1, 2, 3, 4, 5]),
            1,
            0,
            3,
            0,
        )]);

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[2, 3, 4],))
        )
    });
}

#[test]
fn with_subbinary_returns_binary() {
    with_process(|process| {
        let iolist = process.subbinary_from_original(
            process.binary_from_bytes(&[1, 2, 3, 4, 5]),
            1,
            0,
            3,
            0,
        );

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[2, 3, 4],))
        )
    });
}

#[test]
fn with_improper_list_smallint_tail_errors_badarg() {
    with_process(|process| {
        let tail = process.integer(42);
        let iolist = process.improper_list_from_slice(
            &[process.subbinary_from_original(
                process.binary_from_bytes(&[1, 2, 3, 4, 5]),
                1,
                0,
                3,
                0,
            )],
            tail,
        );

        assert_badarg!(
            result(process, iolist),
            format!("iolist ({}) tail ({}) cannot be a byte", iolist, tail)
        )
    });
}

// List elements must be smallint, binary, or lists thereof
#[test]
fn with_atom_in_iolist_errors_badarg() {
    with_process(|process| {
        let element = Atom::str_to_term("foo");
        let iolist = process.list_from_slice(&[element]);

        assert_badarg!(
            result(process, iolist),
            format!(
                "iolist ({}) element ({}) is not a byte, binary, or nested iolist",
                iolist, element
            )
        )
    });
}

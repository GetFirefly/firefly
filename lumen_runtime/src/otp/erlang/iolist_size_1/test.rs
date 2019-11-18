use crate::otp;
use crate::scheduler::with_process;
use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::atom_unchecked;

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

        assert_eq!(
            otp::erlang::iolist_size_1::native(process, iolist),
            Ok(process.integer(4).unwrap())
        )
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

        assert_eq!(
            otp::erlang::iolist_size_1::native(process, iolist),
            Ok(process.integer(10).unwrap())
        )
    });
}

#[test]
fn with_binary_returns_size() {
    with_process(|process| {
        let bin = process.binary_from_bytes(&[1, 2, 3]).unwrap();

        assert_eq!(
            otp::erlang::iolist_size_1::native(process, bin),
            Ok(process.integer(3).unwrap())
        )
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

        assert_eq!(
            otp::erlang::iolist_size_1::native(process, iolist),
            Ok(process.integer(3).unwrap())
        )
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

        assert_eq!(
            otp::erlang::iolist_size_1::native(process, iolist),
            Ok(process.integer(3).unwrap())
        )
    });
}

#[test]
fn with_improper_list_smallint_tail_errors_badarg() {
    with_process(|process| {
        let iolist = process
            .improper_list_from_slice(
                &[process.binary_from_bytes(&[1, 2, 3]).unwrap()],
                process.integer(42).unwrap(),
            )
            .unwrap();

        assert_eq!(
            otp::erlang::iolist_size_1::native(process, iolist),
            Err(badarg!().into())
        )
    });
}

// List elements must be smallint, binary, or lists thereof
#[test]
fn with_atom_in_iolist_errors_badarg() {
    with_process(|process| {
        let iolist = process.list_from_slice(&[atom_unchecked("foo")]).unwrap();

        assert_eq!(
            otp::erlang::iolist_size_1::native(process, iolist),
            Err(badarg!().into())
        )
    });
}

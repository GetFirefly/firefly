use crate::otp;
use crate::scheduler::with_process;

#[test]
fn with_binary_returns_binary_in_list() {
    with_process(|process| {
        let bin = process.binary_from_bytes(&[1, 2, 3]).unwrap();

        assert_eq!(
            otp::erlang::iolist_to_iovec_1::native(process, bin),
            Ok(process
                .list_from_slice(&[process.binary_from_bytes(&[1, 2, 3]).unwrap()])
                .unwrap())
        )
    });
}

#[test]
fn with_mixed_iolists_returns_iovec() {
    with_process(|process| {
        let bin1 = process.binary_from_bytes(&[57, 58, 59]).unwrap();
        let bin2 = process.binary_from_bytes(&[60, 61]).unwrap();
        let bin3 = process.binary_from_bytes(&[6]).unwrap();

        let iolist1 = process
            .improper_list_from_slice(
                &[
                    bin1,
                    process.integer(77).unwrap(),
                    process
                        .list_from_slice(&[
                            process.integer(9).unwrap(),
                            process.integer(8).unwrap(),
                            bin2,
                        ])
                        .unwrap(),
                    process.integer(7).unwrap(),
                ],
                bin3,
            )
            .unwrap();

        let iolist2 = process
            .list_from_slice(&[process.integer(42).unwrap(), process.integer(43).unwrap()])
            .unwrap();

        let iolist_arg = process.list_from_slice(&[iolist1, bin2, iolist2]).unwrap();

        assert_eq!(
            otp::erlang::iolist_to_iovec_1::native(process, iolist_arg),
            Ok(process
                .list_from_slice(&[
                    process
                        .binary_from_bytes(&[57, 58, 59, 77, 9, 8, 60, 61, 7, 6])
                        .unwrap(),
                    process.binary_from_bytes(&[60, 61]).unwrap(),
                    process.binary_from_bytes(&[42, 43]).unwrap()
                ])
                .unwrap())
        )
    });
}

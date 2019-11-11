use crate::otp;
use crate::scheduler::{with_process};

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
            otp::erlang::list_to_binary_1::native(process, iolist),
            Ok(process
                .binary_from_bytes(&[1, 2, 3, 1, 2, 3, 4, 5, 4, 6],)
                .unwrap())
        )
    });
}

#[test]
fn with_binary_returns_binary() {
    with_process(|process| {
        let bin = process.binary_from_bytes(&[1, 2, 3]).unwrap();

        assert_eq!(
            otp::erlang::iolist_to_binary_1::native(process, bin),
            Ok(process
                .binary_from_bytes(&[1, 2, 3],)
                .unwrap())
        )
    });
}
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

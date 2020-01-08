use crate::otp::erlang::date_0;
use crate::otp::erlang::tuple_size_1;
use crate::scheduler::with_process;

#[test]
fn returns_date_as_a_three_tuple() {
    with_process(|process| {
        let date_result = date_0::native(process).unwrap();
        let tuple_size_result = tuple_size_1::native(process, date_result).unwrap();

        assert_eq!(tuple_size_result, process.integer(3).unwrap());
    });
}

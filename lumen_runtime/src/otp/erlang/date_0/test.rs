use crate::otp::erlang::date_0::native;
use crate::scheduler::with_process;

#[test]
fn returns_date_as_a_three_tuple() {
    with_process(|process| {
        let result = native(process);
        assert!(result.is_ok());
    });
}

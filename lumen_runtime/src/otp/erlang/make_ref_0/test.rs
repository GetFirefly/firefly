use crate::otp::erlang::make_ref_0::native;
use crate::scheduler::with_process;

#[test]
fn returns_a_unique_reference() {
    with_process(|process| {
        let first_reference = native(process);
        let second_reference = native(process);

        assert_eq!(first_reference, first_reference);
        assert_ne!(first_reference, second_reference);
        assert_eq!(second_reference, second_reference);
    })
}

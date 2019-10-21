use crate::otp::erlang::monotonic_time_0;
use crate::otp::erlang::subtract_2;
use crate::otp::erlang::system_time_0;
use crate::otp::erlang::time_offset_0;
use crate::scheduler::with_process;

#[test]
fn time_offset_equals_system_time_minus_monotonic_time() {
    with_process(|process| {
        let monotonic_time = monotonic_time_0::native(process).unwrap();
        let system_time = system_time_0::native(process).unwrap();
        let time_offset = time_offset_0::native(process).unwrap();
        let expected_time_offset =
            subtract_2::native(process, system_time, monotonic_time).unwrap();

        println!("monotonic_time is: {:?}", monotonic_time);
        println!("system_time is: {:?}", system_time);
        println!("time_offset is: {:?}", time_offset);
        println!("expected_time_offset is: {:?}", expected_time_offset);

        assert!(time_offset == expected_time_offset);
    });
}

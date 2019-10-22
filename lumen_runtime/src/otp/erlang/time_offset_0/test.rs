use crate::otp::erlang::monotonic_time_0;
use crate::otp::erlang::subtract_2;
use crate::otp::erlang::system_time_0;
use crate::otp::erlang::time_offset_0;
use crate::scheduler::with_process;

const TIME_OFFSET_DELTA_LIMIT: u64 = 2;

#[test]
fn approximately_system_time_minus_monotonic_time() {
    with_process(|process| {
        let monotonic_time = monotonic_time_0::native(process).unwrap();
        let system_time = system_time_0::native(process).unwrap();
        let time_offset = time_offset_0::native(process).unwrap();
        let expected_time_offset =
            subtract_2::native(process, system_time, monotonic_time).unwrap();

        let time_offset_delta =
            subtract_2::native(process, expected_time_offset, time_offset).unwrap();

        assert!(time_offset_delta <= process.integer(TIME_OFFSET_DELTA_LIMIT).unwrap());
    });
}

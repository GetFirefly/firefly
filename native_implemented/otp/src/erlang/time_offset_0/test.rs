use crate::erlang::monotonic_time_0;
use crate::erlang::subtract_2;
use crate::erlang::system_time_0;
use crate::erlang::time_offset_0;
use crate::test::with_process;

const TIME_OFFSET_DELTA_LIMIT: u64 = 20;

#[test]
fn approximately_system_time_minus_monotonic_time() {
    with_process(|process| {
        let monotonic_time = monotonic_time_0::result(process).unwrap();
        let system_time = system_time_0::result(process).unwrap();
        let time_offset = time_offset_0::result(process).unwrap();
        let expected_time_offset =
            subtract_2::result(process, system_time, monotonic_time).unwrap();

        let time_offset_delta =
            subtract_2::result(process, expected_time_offset, time_offset).unwrap();

        assert!(
            time_offset_delta <= process.integer(TIME_OFFSET_DELTA_LIMIT).unwrap(),
            "time_offset_delta ({:?}) <= TIME_OFFSET_DELTA_LIMIT ({:?})",
            time_offset_delta,
            TIME_OFFSET_DELTA_LIMIT
        );
    });
}

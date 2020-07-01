use liblumen_alloc::atom;

use crate::erlang::{
    convert_time_unit_3, monotonic_time_1, subtract_2, system_time_1, time_offset_1,
};
use crate::test::with_process;

const TIME_OFFSET_DELTA_LIMIT_SECONDS: u64 = 2;

#[test]
fn approximately_system_time_minus_monotonic_time_in_seconds() {
    approximately_system_time_minus_monotonic_time_in_unit("second")
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_milliseconds() {
    approximately_system_time_minus_monotonic_time_in_unit("millisecond")
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_microseconds() {
    approximately_system_time_minus_monotonic_time_in_unit("microsecond")
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_nanoseconds() {
    approximately_system_time_minus_monotonic_time_in_unit("nanosecond");
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_native_time_units() {
    approximately_system_time_minus_monotonic_time_in_unit("native");
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_perf_counter_ticks() {
    approximately_system_time_minus_monotonic_time_in_unit("perf_counter");
}

fn approximately_system_time_minus_monotonic_time_in_unit(unit_str: &str) {
    with_process(|process| {
        let unit = atom!(unit_str);
        let monotonic_time = monotonic_time_1::result(process, unit).unwrap();
        let system_time = system_time_1::result(process, unit).unwrap();
        let time_offset = time_offset_1::result(process, unit).unwrap();
        let expected_time_offset =
            subtract_2::result(process, system_time, monotonic_time).unwrap();
        let time_offset_delta =
            subtract_2::result(process, expected_time_offset, time_offset).unwrap();
        let time_offset_delta_limit_seconds =
            process.integer(TIME_OFFSET_DELTA_LIMIT_SECONDS).unwrap();
        let time_offset_delta_limit = convert_time_unit_3::result(
            process,
            time_offset_delta_limit_seconds,
            atom!("seconds"),
            unit,
        )
        .unwrap();

        assert!(
            time_offset_delta <= time_offset_delta_limit,
            "time_offset_delta ({:?}) <= TIME_OFFSET_DELTA_LIMIT ({:?}) in unit ({:?})",
            time_offset_delta,
            time_offset_delta_limit,
            unit
        );
    });
}

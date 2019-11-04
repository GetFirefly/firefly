use crate::otp::erlang::monotonic_time_1;
use crate::otp::erlang::subtract_2;
use crate::otp::erlang::system_time_1;
use crate::otp::erlang::time_offset_1;
use crate::scheduler::with_process;
use liblumen_alloc::erts::term::atom_unchecked;

const TIME_OFFSET_DELTA_LIMIT: u64 = 5;

#[test]
fn approximately_system_time_minus_monotonic_time_in_seconds() {
    with_process(|process| {
        let unit = atom_unchecked("second");

        let monotonic_time = monotonic_time_1::native(process, unit).unwrap();
        let system_time = system_time_1::native(process, unit).unwrap();
        let time_offset = time_offset_1::native(process, unit).unwrap();
        let expected_time_offset =
            subtract_2::native(process, system_time, monotonic_time).unwrap();

        let time_offset_delta =
            subtract_2::native(process, expected_time_offset, time_offset).unwrap();

        assert!(time_offset_delta <= process.integer(TIME_OFFSET_DELTA_LIMIT).unwrap());
    });
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_milliseconds() {
    with_process(|process| {
        let unit = atom_unchecked("millisecond");

        let monotonic_time = monotonic_time_1::native(process, unit).unwrap();
        let system_time = system_time_1::native(process, unit).unwrap();
        let time_offset = time_offset_1::native(process, unit).unwrap();
        let expected_time_offset =
            subtract_2::native(process, system_time, monotonic_time).unwrap();

        let time_offset_delta =
            subtract_2::native(process, expected_time_offset, time_offset).unwrap();

        assert!(time_offset_delta <= process.integer(TIME_OFFSET_DELTA_LIMIT).unwrap());
    });
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_microseconds() {
    with_process(|process| {
        let unit = atom_unchecked("microsecond");

        let monotonic_time = monotonic_time_1::native(process, unit).unwrap();
        let system_time = system_time_1::native(process, unit).unwrap();
        let time_offset = time_offset_1::native(process, unit).unwrap();
        let expected_time_offset =
            subtract_2::native(process, system_time, monotonic_time).unwrap();

        let time_offset_delta =
            subtract_2::native(process, expected_time_offset, time_offset).unwrap();

        assert!(time_offset_delta <= process.integer(TIME_OFFSET_DELTA_LIMIT).unwrap());
    });
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_nanoseconds() {
    with_process(|process| {
        let unit = atom_unchecked("nanosecond");

        let monotonic_time = monotonic_time_1::native(process, unit).unwrap();
        let system_time = system_time_1::native(process, unit).unwrap();
        let time_offset = time_offset_1::native(process, unit).unwrap();
        let expected_time_offset =
            subtract_2::native(process, system_time, monotonic_time).unwrap();

        let time_offset_delta =
            subtract_2::native(process, expected_time_offset, time_offset).unwrap();

        assert!(time_offset_delta <= process.integer(TIME_OFFSET_DELTA_LIMIT).unwrap());
    });
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_native_time_units() {
    with_process(|process| {
        let unit = atom_unchecked("native");

        let monotonic_time = monotonic_time_1::native(process, unit).unwrap();
        let system_time = system_time_1::native(process, unit).unwrap();
        let time_offset = time_offset_1::native(process, unit).unwrap();
        let expected_time_offset =
            subtract_2::native(process, system_time, monotonic_time).unwrap();

        let time_offset_delta =
            subtract_2::native(process, expected_time_offset, time_offset).unwrap();

        assert!(time_offset_delta <= process.integer(TIME_OFFSET_DELTA_LIMIT).unwrap());
    });
}

#[test]
fn approximately_system_time_minus_monotonic_time_in_perf_counter_ticks() {
    with_process(|process| {
        let unit = atom_unchecked("perf_counter");

        let monotonic_time = monotonic_time_1::native(process, unit).unwrap();
        let system_time = system_time_1::native(process, unit).unwrap();
        let time_offset = time_offset_1::native(process, unit).unwrap();
        let expected_time_offset =
            subtract_2::native(process, system_time, monotonic_time).unwrap();

        let time_offset_delta =
            subtract_2::native(process, expected_time_offset, time_offset).unwrap();

        assert!(time_offset_delta <= process.integer(TIME_OFFSET_DELTA_LIMIT).unwrap());
    });
}

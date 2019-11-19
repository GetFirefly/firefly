use crate::otp::erlang::{add_2, multiply_2, subtract_2, system_time_1, timestamp_0, tuple_size_1};
use crate::scheduler::with_process;
use liblumen_alloc::erts::term::prelude::*;
use std::convert::TryInto;

const DELTA_LIMIT_MICROSECONDS: u64 = 5;

#[test]
fn returns_a_three_element_tuple() {
    with_process(|process| {
        let timestamp = timestamp_0::native(process).unwrap();

        let tuple_size_result = tuple_size_1::native(process, timestamp).unwrap();

        assert!(tuple_size_result == process.integer(3).unwrap());
    });
}

#[test]
fn approximately_system_time() {
    // Take the result from timestamp and
    // compare it to system time
    with_process(|process| {
        let unit = Atom::str_to_term("microsecond");

        let system_time = system_time_1::native(process, unit).unwrap();

        let timestamp = timestamp_0::native(process).unwrap();

        let timestamp_tuple: Boxed<Tuple> = timestamp.try_into().unwrap();

        let megasecs = multiply_2::native(
            process,
            timestamp_tuple.get_element(0).unwrap(),
            process.integer(1000000000000 as usize).unwrap(),
        )
        .unwrap();
        let secs = multiply_2::native(
            process,
            timestamp_tuple.get_element(1).unwrap(),
            process.integer(1000000).unwrap(),
        )
        .unwrap();

        let microsecs = timestamp_tuple.get_element(2).unwrap();

        let system_time_from_timestamp = add_2::native(
            process,
            add_2::native(process, megasecs, secs).unwrap(),
            microsecs,
        )
        .unwrap();

        let delta_limit_microseconds = process.integer(DELTA_LIMIT_MICROSECONDS).unwrap();

        let delta = subtract_2::native(process, system_time_from_timestamp, system_time).unwrap();

        assert!(delta < delta_limit_microseconds);
    });
}

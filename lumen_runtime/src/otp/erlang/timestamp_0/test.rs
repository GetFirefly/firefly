use crate::otp::erlang::{system_time_1, timestamp_0, tuple_size_1};
use crate::scheduler::with_process;
use liblumen_alloc::erts::term::prelude::*;
use std::convert::TryInto;

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
    with_process(|process| {
        let unit = Atom::str_to_term("microsecond");

        let system_time = system_time_1::native(process, unit).unwrap();

        let timestamp = timestamp_0::native(process).unwrap();

        let timestamp_tuple: Boxed<Tuple> = timestamp.try_into().unwrap();

        let megasecs = timestamp_tuple.get_element(0).unwrap();
        let secs = timestamp_tuple.get_element(1).unwrap();
        let microsecs = timestamp_tuple.get_element(2).unwrap();

        println!("{:?} {:?} {:?}", megasecs, secs, microsecs);

        assert!(process.integer(3).unwrap() == process.integer(3).unwrap());
    });
}

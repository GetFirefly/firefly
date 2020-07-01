use crate::erlang::monotonic_time_0::result;
use crate::test::with_process;

use crate::runtime::time::monotonic;

#[test]
fn increases_after_2_native_time_units() {
    with_process(|process| {
        let start_time_in_milliseconds = monotonic::freeze_time_in_milliseconds();

        let first = result(process).unwrap();

        monotonic::freeze_at_time_in_milliseconds(start_time_in_milliseconds + 2);

        let second = result(process).unwrap();

        assert!(first < second);
    });
}

use crate::erlang::monotonic_time_0::result;
use crate::test::with_process;

use crate::runtime::time::monotonic;

#[test]
fn increases_after_2_native_time_units() {
    with_process(|process| {
        let start_monotonic = monotonic::freeze();

        let first = result(process);

        monotonic::freeze_at(start_monotonic + Milliseconds(2));

        let second = result(process);

        assert!(first < second);
    });
}

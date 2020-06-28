use std::thread;
use std::time::Duration;

use crate::erlang::system_time_0::result;
use crate::test::with_process;

#[test]
fn increases_after_2_native_time_units() {
    with_process(|process| {
        let first = result(process).unwrap();

        thread::sleep(Duration::from_millis(2));

        let second = result(process).unwrap();

        assert!(first < second);
    });
}

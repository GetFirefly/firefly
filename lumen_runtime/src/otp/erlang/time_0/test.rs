use std::thread;
use std::time::Duration;

use crate::otp::erlang::time_0::native;
use crate::scheduler::with_process;

#[test]
fn increases_after_2_seconds() {
    with_process(|process| {
        let first = native(process).unwrap();

        thread::sleep(Duration::from_secs(2));

        let second = native(process).unwrap();

        assert!(first < second);
    });
}

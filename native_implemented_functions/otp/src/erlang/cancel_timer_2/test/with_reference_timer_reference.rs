use super::*;

mod with_empty_list_options;
mod with_list_options;

fn in_same_thread_without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message(
    options: fn(&Process) -> Term,
) {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        let start_time_in_milliseconds = freeze_timeout();
        let half_milliseconds = milliseconds / 2;
        freeze_at_timeout(start_time_in_milliseconds + half_milliseconds + 1);

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        let first_result = result(process, timer_reference, options(process));

        assert!(first_result.is_ok());

        let milliseconds_remaining = first_result.unwrap();

        assert!(milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < milliseconds_remaining);
        assert!(milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(false.into())
        );

        freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(false.into())
        );
    })
}

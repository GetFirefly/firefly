#[path = "with_local_reference/with_timer.rs"]
mod with_timer;

test_stdout!(without_timer_returns_ok, "false\n");

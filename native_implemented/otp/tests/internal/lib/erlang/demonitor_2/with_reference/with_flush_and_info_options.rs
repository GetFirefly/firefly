#[path = "with_flush_and_info_options/with_monitor.rs"]
mod with_monitor;

test_stdout!(without_monitor_returns_false, "false\n");

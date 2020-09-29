#[path = "with_flush_option/with_monitor.rs"]
mod with_monitor;

test_stdout!(without_monitor_returns_true, "true\n");

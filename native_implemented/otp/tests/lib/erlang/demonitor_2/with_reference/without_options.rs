#[path = "without_options/with_monitor.rs"]
pub mod with_monitor;

test_stdout!(without_monitor_returns_true, "true\n");

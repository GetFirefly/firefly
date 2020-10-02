#[path = "without_self/with_process.rs"]
mod with_process;

test_stdout!(without_process_returns_false, "false\n");

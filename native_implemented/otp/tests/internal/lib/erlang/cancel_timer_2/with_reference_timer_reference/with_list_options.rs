#[path = "with_list_options/with_async_false.rs"]
mod with_async_false;
#[path = "with_list_options/with_async_true.rs"]
mod with_async_true;
#[path = "with_list_options/without_async.rs"]
mod without_async;

test_stdout!(with_invalid_option, "{caught, error, badarg}\n");

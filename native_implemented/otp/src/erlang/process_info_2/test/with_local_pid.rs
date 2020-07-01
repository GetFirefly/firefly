mod with_registered_name;

use super::*;

#[test]
fn without_supported_item_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&unsupported_item_atom(), |item| {
                let pid = arc_process.pid_term();
                prop_assert_badarg!(
                    result(&arc_process, pid, item),
                    "supported items are backtrace, binary, catchlevel, current_function, \
                     current_location, current_stacktrace, dictionary, error_handler, \
                     garbage_collection, garbage_collection_info, group_leader, heap_size, \
                     initial_call, links, last_calls, memory, message_queue_len, messages, \
                     min_heap_size, min_bin_vheap_size, monitored_by, monitors, \
                     message_queue_data, priority, reductions, registered_name, \
                     sequential_trace_token, stack_size, status, suspending, \
                     total_heap_size, trace, trap_exit"
                );

                Ok(())
            })
            .unwrap();
    });
}

fn unsupported_item_atom() -> BoxedStrategy<Term> {
    strategy::atom()
        .prop_filter("Item cannot be supported", |atom| match atom.name() {
            "registered_name" => false,
            _ => true,
        })
        .prop_map(|atom| atom.encode().unwrap())
        .boxed()
}

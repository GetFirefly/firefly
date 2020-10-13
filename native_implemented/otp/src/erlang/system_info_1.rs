use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:system_info/1)]
pub fn result(item: Term) -> exception::Result<Term> {
    match item.decode().unwrap() {
        TypedTerm::Atom(atom) => match atom.name() {
            "alloc_util_allocators" => unimplemented!(),
            "allocated_areas" => unimplemented!(),
            "allocator" => unimplemented!(),
            "atom_count" => unimplemented!(),
            "atom_limit" => unimplemented!(),
            "build_type" => unimplemented!(),
            "c_compiler_used" => unimplemented!(),
            "check_io" => unimplemented!(),
            "compat_rel" => unimplemented!(),
            "cpu_quota" => unimplemented!(),
            "cpu_topology" => unimplemented!(),
            "creation" => unimplemented!(),
            "debug_compiled" => unimplemented!(),
            "delayed_node_table_gc" => unimplemented!(),
            "dirty_cpu_schedulers" => unimplemented!(),
            "dirty_cpu_schedulers_online" => unimplemented!(),
            "dirty_io_schedulers" => unimplemented!(),
            "dist" => unimplemented!(),
            "dist_buf_busy_limit" => unimplemented!(),
            "dist_ctrl" => unimplemented!(),
            "driver_version" => unimplemented!(),
            "dynamic_trace" => unimplemented!(),
            "dynamic_trace_probes" => unimplemented!(),
            "elib_malloc" => unimplemented!(),
            "end_time" => unimplemented!(),
            "ets_count" => unimplemented!(),
            "ets_limit" => unimplemented!(),
            "fullsweep_after" => unimplemented!(),
            "garbage_collection" => unimplemented!(),
            "heap_sizes" => unimplemented!(),
            "heap_type" => unimplemented!(),
            "info" => unimplemented!(),
            "kernel_poll" => unimplemented!(),
            "loaded" => unimplemented!(),
            "logic_processors" => unimplemented!(),
            "logic_processors_available" => unimplemented!(),
            "logical_processors_online" => unimplemented!(),
            "machine" => unimplemented!(),
            "max_heap_size" => unimplemented!(),
            "message_queue_data" => unimplemented!(),
            "min_bin_vheap_size" => unimplemented!(),
            "min_heap_size" => unimplemented!(),
            "modified_timing_level" => unimplemented!(),
            "multi_scheduling" => unimplemented!(),
            "multi_scheduling_blockers" => unimplemented!(),
            "nif_version" => unimplemented!(),
            "normal_multi_scheduling_blockers" => unimplemented!(),
            "os_monotonic_time_source" => unimplemented!(),
            "os_system_time_source" => unimplemented!(),
            "otp_release" => unimplemented!(),
            "port_count" => unimplemented!(),
            "port_limit" => unimplemented!(),
            "port_parallelism" => unimplemented!(),
            "process_count" => unimplemented!(),
            "process_limit" => unimplemented!(),
            "procs" => unimplemented!(),
            "scheduler_bind_type" => unimplemented!(),
            "scheduler_bindings" => unimplemented!(),
            "scheduler_id" => unimplemented!(),
            "schedulers" => unimplemented!(),
            "schedulers_online" => unimplemented!(),
            "sequential_tracer" => unimplemented!(),
            "smp_support" => unimplemented!(),
            "start_time" => unimplemented!(),
            "system_architecture" => unimplemented!(),
            "system_logger" => unimplemented!(),
            "system_version" => unimplemented!(),
            "thread_pool_size" => unimplemented!(),
            "threads" => unimplemented!(),
            "time_correction" => unimplemented!(),
            "time_offset" => unimplemented!(),
            "time_warp_mode" => unimplemented!(),
            "tolerant_timeofday" => unimplemented!(),
            "trace_control_word" => unimplemented!(),
            "update_cpu_info" => unimplemented!(),
            "version" => unimplemented!(),
            "wordsize" => unimplemented!(),
            _ => Err(anyhow!(
                "item ({}) is not a supported atom ({})",
                item,
                SUPPORTED_ATOMS
            )
            .into()),
        },
        TypedTerm::Tuple(boxed_tuple) => {
            if boxed_tuple.len() == 2 {
                let tag = boxed_tuple[0];

                match tag.decode().unwrap() {
                    TypedTerm::Atom(tag_atom) => match tag_atom.name() {
                        "allocator" => unimplemented!(),
                        "allocator_sizes" => unimplemented!(),
                        "cpu_topology" => unimplemented!(),
                        "wordsize" => unimplemented!(),
                        _ => item_is_not_supported_tuple(item),
                    },
                    _ => item_is_not_supported_tuple(item),
                }
            } else {
                item_is_not_supported_tuple(item)
            }
        }
        _ => Err(anyhow!(
            "item ({}) is not either an atom ({}) or tuple ({})",
            item,
            SUPPORTED_ATOMS,
            SUPPORTED_TUPLES
        )
        .into()),
    }
}

const SUPPORTED_ATOMS: &'static str = "`allocated_areas`, `allocator`, \
                 `alloc_util_allocators`, `elib_malloc`, `cpu_topology`, `logic_processors`, \
                 `logic_processors_available`, `logical_processors_online`, \
                 `cpu_quota`, `update_cpu_info`, `fullsweep_after`, `garbage_collection`, \
                 `heap_sizes`, `heap_type`, `max_heap_size`, `message_queue_data`, `min_heap_size` \
                 `min_bin_vheap_size`, `procs`, `atom_count`, `atom_limit`, `ets_count`, \
                 `ets_limit`, `port_count`, `port_limit`, `process_count`, `process_limit`, \
                 `end_time`, `os_monotonic_time_source`, `os_system_time_source`, `start_time` \
                 `time_correction`, `time_offset`, `time_warp_mode`, `tolerant_timeofday` \
                 `dirty_cpu_schedulers`, `dirty_cpu_schedulers_online`, `dirty_io_schedulers`, \
                 `multi_scheduling`, `multi_scheduling_blockers`, \
                 `normal_multi_scheduling_blockers`, `scheduler_bind_type`, `scheduler_bindings`, \
                 `scheduler_id`, `schedulers`, `schedulers_online`, `smp_support`, `threads`, \
                 `thread_pool_size`, `creation`, `delayed_node_table_gc`, `dist`, \
                 `dist_buf_busy_limit`, `dist_ctrl`, `build_type`, `c_compiler_used`, `check_io`, \
                 `compat_rel`, `debug_compiled`, `driver_version`, `dynamic_trace`, \
                 `dynamic_trace_probes`, `info`, `kernel_poll`, `loaded`, `machine`, \
                 `modified_timing_level`, `nif_version`, `otp_release`, `port_parallelism`, \
                 `sequential_tracer`, \
                 `system_architecture`, `system_logger`, `system_version`, `trace_control_word`, \
                 `version`, or `wordsize`";

const SUPPORTED_TUPLES: &'static str = "`{allocator, Alloc}`, `{allocator_sizes, Alloc}`, \
          `{cpu_topology, defined | detected | used}`, or `{wordsize, internal | external}`";

fn item_is_not_supported_tuple(item: Term) -> exception::Result<Term> {
    Err(anyhow!(
        "item ({}) is not a supported tuple ({})",
        item,
        SUPPORTED_TUPLES
    )
    .into())
}

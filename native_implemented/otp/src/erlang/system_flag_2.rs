use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:system_flag/2)]
pub fn result(flag: Term, _value: Term) -> Result<Term, NonNull<ErlangException>> {
    let flag_atom = term_try_into_atom!(flag)?;

    match flag_atom.as_str() {
        "backtrace_depth" => unimplemented!(),
        "cpu_topology" => unimplemented!(),
        "dirty_cpu_schedulers_online" => unimplemented!(),
        "erts_alloc" => unimplemented!(),
        "fullsweep_after" => unimplemented!(),
        "microstate_accounting" => unimplemented!(),
        "min_heap_size" => unimplemented!(),
        "min_bin_vheap_size" => unimplemented!(),
        "max_heap_size" => unimplemented!(),
        "multi_scheduling" => unimplemented!(),
        "scheduler_bind_type" => unimplemented!(),
        "schedulers_online" => unimplemented!(),
        "system_logger" => unimplemented!(),
        "trace_control_word" => unimplemented!(),
        "time_offset" => unimplemented!(),
        _ => Err(anyhow!(
            "flag ({}) is not supported (backtrace_depth, cpu_topology, \
             dirty_cpu_schedulers_online, erts_alloc, fullsweep_after, microstate_accounting, \
             min_heap_size, min_bin_vheap_size, max_heap_size, multi_scheduling, \
             scheduler_bind_type, schedulers_online, system_logger, trace_control_word, \
             time_offset)"
        )
        .into()),
    }
}

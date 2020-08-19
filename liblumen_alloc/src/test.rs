use crate::erts::process::ffi::ProcessSignal;

#[export_name = "__lumen_process_signal"]
#[thread_local]
static mut PROCESS_SIGNAL: ProcessSignal = ProcessSignal::None;

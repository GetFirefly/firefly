use alloc::sync::Arc;

use liblumen_alloc::erts::process::ProcessControlBlock;

pub mod queues;

/// What to run
pub enum Run {
    /// Run the process now
    Now(Arc<ProcessControlBlock>),
    /// There was a process in the queue, but it needs to be delayed because it is `Priority::Low`
    /// and hadn't been delayed enough yet.  Ask the `RunQueue` again for another process.
    /// -- https://github.com/erlang/otp/blob/fe2b1323a3866ed0a9712e9d12e1f8f84793ec47/erts/emulator/beam/erl_process.c#L9601-L9606
    Delayed,
    /// There are no processes in the run queue, do other work
    None,
}

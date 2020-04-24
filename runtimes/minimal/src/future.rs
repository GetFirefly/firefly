use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{exception, Arity, Process};

pub use lumen_rt_core::future::*;
use lumen_rt_core::process::spawn::Options;
use lumen_rt_core::registry;
use lumen_rt_core::time::Milliseconds;

use crate::process::runnable;
use crate::scheduler;

pub fn run_until_ready<PlaceFrameWithArguments>(
    options: Options,
    place_frame_with_arguments: PlaceFrameWithArguments,
    timeout: Milliseconds,
) -> Result<Ready, NotReady>
where
    PlaceFrameWithArguments: Fn(&Process) -> exception::Result<()>,
{
    assert!(!options.link);
    assert!(!options.monitor);

    let spawned = spawn(options, place_frame_with_arguments)?;

    spawned.run_until_ready(timeout)
}

fn function() -> Atom {
    Atom::from_str("future")
}

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen")
}

const ARITY: Arity = 0;

fn spawn<PlaceFrameWithArguments>(
    options: Options,
    place_frame_with_arguments: PlaceFrameWithArguments,
) -> exception::Result<Spawned>
where
    PlaceFrameWithArguments: Fn(&Process) -> exception::Result<()>,
{
    let parent_process = None;
    let process = options.spawn(parent_process, module(), function(), ARITY)?;
    runnable(&process);
    let arc_mutex_future = Future::stack_push(&process)?;

    place_frame_with_arguments(&process)?;

    let arc_process = scheduler::current().schedule(process);
    registry::put_pid_to_process(&arc_process);

    Ok(Spawned {
        arc_process,
        arc_mutex_future,
    })
}

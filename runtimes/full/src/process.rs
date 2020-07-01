mod init;
mod out_of_code;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::{self, Frame, FrameWithArguments, Process};

pub use lumen_rt_core::process::{current_process, monitor, spawn};

#[export_name = "lumen_rt_process_runnable"]
pub fn runnable<'a>(
    process: &Process,
    frames_with_arguments_fn: Box<
        dyn FnOnce(&Process) -> AllocResult<Vec<FrameWithArguments>> + 'a,
    >,
) -> AllocResult<()> {
    process.runnable(|process| {
        let mut frames_with_arguments = frames_with_arguments_fn(process)?;

        frames_with_arguments.push(out_of_code::frame().with_arguments(false, &[]));

        for FrameWithArguments {
            frame, arguments, ..
        } in frames_with_arguments.into_iter().rev()
        {
            process.stack_push_slice(&arguments)?;
            process.frames.lock().push(frame.clone());
        }

        Ok(())
    })
}

pub fn init(minimum_heap_size: usize) -> AllocResult<Process> {
    let module_function_arity = init::module_function_arity();

    let heap_size = process::alloc::next_heap_size(minimum_heap_size);
    let heap = process::alloc::heap(heap_size)?;

    let process = Process::new(
        Default::default(),
        None,
        module_function_arity,
        heap,
        heap_size,
    );

    runnable(
        &process,
        Box::new(move |_| {
            let frame = Frame::new(module_function_arity, init::NATIVE);

            Ok(vec![frame.with_arguments(false, &[])])
        }),
    )?;

    Ok(process)
}

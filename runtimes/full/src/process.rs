use alloc::sync::Arc;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::{self, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_rt_core::code;
pub use lumen_rt_core::process::{monitor, spawn};

pub fn init(minimum_heap_size: usize) -> AllocResult<Process> {
    let init = Atom::from_str("init");
    let start = Atom::from_str("start");
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: init,
        function: start,
        arity: 0,
    });

    let heap_size = process::alloc::next_heap_size(minimum_heap_size);
    let heap = process::alloc::heap(heap_size)?;

    let process = Process::new(
        Default::default(),
        None,
        Arc::clone(&module_function_arity),
        heap,
        heap_size,
    );

    let frame = Frame::new(module_function_arity, code::wait);
    process.push_frame(frame);

    Ok(process)
}

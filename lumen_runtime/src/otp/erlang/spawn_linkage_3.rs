use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::{default_heap, ProcessControlBlock};
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::ModuleFunctionArity;

use crate::otp::erlang::{spawn_3, spawn_link_3};
use crate::process::Linkage;
use crate::scheduler::Scheduler;

pub(in crate::otp::erlang) fn native(
    process_control_block: &ProcessControlBlock,
    linkage: Linkage,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result {
    let module_atom: Atom = module.try_into()?;
    let function_atom: Atom = function.try_into()?;

    if arguments.is_proper_list() {
        let (heap, heap_size) = default_heap()?;
        let arc_process = Scheduler::spawn_linkage_apply_3(
            process_control_block,
            linkage,
            module_atom,
            function_atom,
            arguments,
            heap,
            heap_size,
        )?;

        Ok(arc_process.pid_term())
    } else {
        Err(badarg!().into())
    }
}

pub(in crate::otp::erlang) fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    linkage: Linkage,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result<(), Alloc> {
    process.stack_push(arguments)?;
    process.stack_push(function)?;
    process.stack_push(module)?;
    process.place_frame(frame(linkage), placement);

    Ok(())
}

// Private

fn code(linkage: Linkage) -> Code {
    match linkage {
        Linkage::None => spawn_3::code,
        Linkage::Monitor => unimplemented!(),
        Linkage::Link => spawn_link_3::code,
    }
}

fn frame(linkage: Linkage) -> Frame {
    Frame::new(module_function_arity(linkage), code(linkage))
}

fn function(linkage: Linkage) -> Atom {
    let s = match linkage {
        Linkage::None => "spawn",
        Linkage::Monitor => "spawn_monitor",
        Linkage::Link => "spawn_link",
    };

    Atom::try_from_str(s).unwrap()
}

fn module_function_arity(linkage: Linkage) -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(linkage),
        arity: 3,
    })
}

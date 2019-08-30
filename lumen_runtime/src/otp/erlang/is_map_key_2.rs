use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::ModuleFunctionArity;

use crate::otp::maps;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    key: Term,
    map: Term,
) -> Result<(), Alloc> {
    process.stack_push(map)?;
    process.stack_push(key)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn frame() -> Frame {
    Frame::new(module_function_arity(), maps::is_key_2::code)
}

fn function() -> Atom {
    Atom::try_from_str("is_map_key").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

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

const ARITY: Arity = 2;

fn frame() -> Frame {
    Frame::new(
        super::module(),
        function(),
        ARITY,
        maps::is_key_2::LOCATION,
        maps::is_key_2::code,
    )
}

fn function() -> Atom {
    Atom::try_from_str("is_map_key").unwrap()
}

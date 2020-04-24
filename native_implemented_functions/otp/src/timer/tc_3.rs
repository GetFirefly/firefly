mod label_1;
mod label_2;
mod label_3;
mod label_4;
mod label_5;

use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{FrameWithArguments, ModuleFunctionArity};

use crate::erlang::monotonic_time_0;
use crate::runtime::process::current_process;

pub fn frame_with_arguments(module: Term, function: Term, arguments: Term) -> FrameWithArguments {
    label_frame_with_arguments(NATIVE, false, &[module, function, arguments])
}

// Private

const NATIVE: Native = Native::Three(native);

/// ```elixir
/// def tc(module, function, arguments) do
///   before = :erlang.monotonic_time()
///   value = apply(module, function, arguments)
///   after = :erlang.monotonic_time()
///   duration = after - before
///   time = :erlang.convert_time_unit(duration, :native, :microsecond)
///   {time, value}
/// end
/// ```
extern "C" fn native(module: Term, function: Term, arguments: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    arc_process
        .queue_frame_with_arguments(label_1::frame_with_arguments(module, function, arguments));
    arc_process.queue_frame_with_arguments(monotonic_time_0::frame().with_arguments(false, &[]));

    Term::NONE
}

fn frame(native: Native) -> Frame {
    Frame::new(module_function_arity(), native)
}

fn label_frame_with_arguments(
    native: Native,
    uses_returned: bool,
    arguments: &[Term],
) -> FrameWithArguments {
    frame(native).with_arguments(uses_returned, arguments)
}

fn function() -> Atom {
    Atom::try_from_str("t3").unwrap()
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    }
}

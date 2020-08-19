use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:get_keys/0)]
pub fn result(process: &Process) -> Term {
    process.get_keys()
}

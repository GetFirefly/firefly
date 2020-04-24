use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

extern "C" {
    #[link_name = "__lumen_builtin_yield"]
    fn builtin_yield() -> bool;
}

#[native_implemented_function(loop/0)]
fn result(process: &Process) -> Term {
    loop {
        unsafe {
            process.wait();
            builtin_yield();
        }
    }
}

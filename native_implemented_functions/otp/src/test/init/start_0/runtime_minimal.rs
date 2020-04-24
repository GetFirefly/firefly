use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

extern "C" {
    #[link_name = "__lumen_builtin_yield"]
    fn builtin_yield() -> bool;
}

#[native_implemented_function(start/0)]
fn result() -> Term {
    loop {
        crate::runtime::process::current_process().wait();

        unsafe {
            builtin_yield();
        }
    }
}

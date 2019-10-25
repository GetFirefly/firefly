#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::prelude::Term;(

use lumen_runtime_macros::native_implemented_function;

/// Distribution is not supported at this time.  Always returns `false`.
#[native_implemented_function(is_alive/0)]
pub fn native() -> Term {
    false.into()
}

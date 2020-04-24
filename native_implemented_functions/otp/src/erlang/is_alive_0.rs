#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

/// Distribution is not supported at this time.  Always returns `false`.
#[native_implemented_function(is_alive/0)]
pub fn result() -> Term {
    false.into()
}

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::process::replace_log_exit;

#[native_implemented::function(lumen:log_exit/1)]
fn result(log_exit: Term) -> exception::Result<Term> {
    let boolean_bool: bool = term_try_into_bool!(log_exit)?;

    Ok(replace_log_exit(boolean_bool).into())
}

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

use crate::runtime::process::replace_log_exit;

#[native_implemented::function(lumen:log_exit/1)]
fn result(log_exit: Term) -> Result<Term, NonNull<ErlangException>> {
    let boolean_bool: bool = term_try_into_bool!(log_exit)?;

    Ok(replace_log_exit(boolean_bool).into())
}

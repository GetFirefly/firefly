#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:put/2)]
pub fn result(process: &Process, key: Term, value: Term) -> Term {
    process.put(key, value)
}

use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:erase/1)]
pub fn result(process: &Process, key: Term) -> Term {
    process.erase_value_from_key(key)
}

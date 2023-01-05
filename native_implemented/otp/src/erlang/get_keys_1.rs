use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:get_keys/1)]
pub fn result(process: &Process, value: Term) -> Term {
    process.get_keys_from_value(value)
}

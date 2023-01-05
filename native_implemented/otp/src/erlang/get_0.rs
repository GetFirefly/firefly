use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:get/0)]
pub fn result(process: &Process) -> Term {
    process.get_entries()
}

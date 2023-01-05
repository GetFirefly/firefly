use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(test:start/0)]
fn result(process: &Process) -> Term {
    process.wait();

    Term::NONE
}

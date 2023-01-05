use firefly_rt::process::Process;
use firefly_rt::term::Term;

pub fn anonymous_closure(process: &Process) -> Term {
    process.anonymous_closure_with_env_from_slice(
        super::module(),
        INDEX,
        OLD_UNIQUE,
        UNIQUE,
        ARITY,
        CLOSURE_NATIVE,
        process.pid().into(),
        &[],
    )
}

const INDEX: Index = 1;
const OLD_UNIQUE: OldUnique = 2;
const UNIQUE: Unique = [
    0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01,
];

#[native_implemented::function(test:1-2-23456789ABCDEF0123456789ABCDEF01/1)]
fn result(argument: Term) -> Term {
    argument
}

pub mod tc_3;

pub mod cancel;
pub mod read;
pub mod start;

fn module() -> Atom {
    Atom::try_from_str("timer").unwrap()
}

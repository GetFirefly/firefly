#[path = "./without_body/label_1.rs"]
pub mod label_1;

use super::*;

fn function() -> Atom {
    Atom::try_from_str("body_1_without_body").unwrap()
}

fn module() -> Atom {
    Atom::try_from_str("Lumen.Web.DocumentTest").unwrap()
}

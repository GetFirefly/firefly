#[path = "with_body/label_1.rs"]
pub mod label_1;
#[path = "with_body/label_2.rs"]
pub mod label_2;

use super::*;

fn function() -> Atom {
    Atom::try_from_str("body_1_with_body").unwrap()
}

fn module() -> Atom {
    Atom::try_from_str("Lumen.Web.DocumentTest").unwrap()
}

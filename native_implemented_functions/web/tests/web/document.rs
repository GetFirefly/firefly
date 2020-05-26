#[path = "document/body_1.rs"]
mod body_1;
#[path = "document/new_0.rs"]
mod new_0;

use super::*;

fn module() -> Atom {
    Atom::from_str("Lumen.Web.DocumentTest")
}

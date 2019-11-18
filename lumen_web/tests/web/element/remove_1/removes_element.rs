#[path = "removes_element/label_1.rs"]
pub mod label_1;
#[path = "removes_element/label_2.rs"]
pub mod label_2;
#[path = "removes_element/label_3.rs"]
pub mod label_3;
#[path = "removes_element/label_4.rs"]
pub mod label_4;
#[path = "removes_element/label_5.rs"]
pub mod label_5;

use liblumen_alloc::erts::term::prelude::Atom;

fn function() -> Atom {
    Atom::try_from_str("remove_1_removes_element").unwrap()
}

fn module() -> Atom {
    Atom::try_from_str("Lumen.Web.ElementTest").unwrap()
}

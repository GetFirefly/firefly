#[path = "./with_nil_reference_child_appends_new_child/label_1.rs"]
pub mod label_1;
#[path = "./with_nil_reference_child_appends_new_child/label_2.rs"]
pub mod label_2;
#[path = "./with_nil_reference_child_appends_new_child/label_3.rs"]
pub mod label_3;
#[path = "./with_nil_reference_child_appends_new_child/label_4.rs"]
pub mod label_4;
#[path = "./with_nil_reference_child_appends_new_child/label_5.rs"]
pub mod label_5;
#[path = "./with_nil_reference_child_appends_new_child/label_6.rs"]
pub mod label_6;

use liblumen_alloc::erts::term::prelude::Atom;

fn function() -> Atom {
    Atom::try_from_str("insert_before_3_with_nil_reference_child_appends_new_child").unwrap()
}

fn module() -> Atom {
    Atom::try_from_str("Lumen.Web.NodeTest").unwrap()
}

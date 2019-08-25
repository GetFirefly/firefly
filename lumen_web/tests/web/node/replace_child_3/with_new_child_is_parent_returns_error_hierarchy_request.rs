#[path = "./with_new_child_is_parent_returns_error_hierarchy_request/label_1.rs"]
pub mod label_1;
#[path = "./with_new_child_is_parent_returns_error_hierarchy_request/label_2.rs"]
pub mod label_2;
#[path = "./with_new_child_is_parent_returns_error_hierarchy_request/label_3.rs"]
pub mod label_3;
#[path = "./with_new_child_is_parent_returns_error_hierarchy_request/label_4.rs"]
pub mod label_4;

use liblumen_alloc::erts::term::Atom;

fn function() -> Atom {
    Atom::try_from_str("replace_child_3_with_new_child_is_parent_returns_error_hierarchy_request")
        .unwrap()
}

fn module() -> Atom {
    Atom::try_from_str("Lumen.Web.NodeTest").unwrap()
}

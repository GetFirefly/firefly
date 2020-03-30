use super::*;

mod with_registered_name;

#[test]
fn without_registered_name_errors_badarg() {
    let name = registered_name();

    assert_badarg!(native(name), format!("name ({}) was not registered", name));
}

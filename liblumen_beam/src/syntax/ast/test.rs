use crate::syntax::ast::*;

#[test]
fn it_works() {
    AST::from_beam_file("tests/testdata/ast/test.beam")
        .map_err(|err| {
            println!("[ERROR] {}", err);
            "Failed"
        })
        .unwrap();
}

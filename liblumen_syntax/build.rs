extern crate lalrpop;

fn main() {
    lalrpop::Configuration::new()
        .use_cargo_dir_conventions()
        .process_file("src/parser/grammar.lalrpop")
        .unwrap();

    println!("cargo:rerun-if-changed=src/parser/grammar.lalrpop");
}

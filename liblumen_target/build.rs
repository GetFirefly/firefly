use std::env;

fn main() {
    let triple = env::var("TARGET").expect("TARGET");
    println!("cargo:rustc-env=TARGET={}", triple);
}

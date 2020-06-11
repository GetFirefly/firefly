use std::env;
use std::path::PathBuf;

fn main() {
    let mut outdir = PathBuf::from(env::var("OUT_DIR").unwrap());
    assert!(outdir.pop());
    assert!(outdir.pop());
    assert!(outdir.pop());
    println!("cargo:rustc-link-search={}", outdir.display());
}

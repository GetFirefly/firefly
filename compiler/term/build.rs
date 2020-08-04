extern crate which;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const ENV_LLVM_PREFIX: &'static str = "LLVM_PREFIX";

fn main() {
    // Emit custom cfg types:
    //     cargo:rustc-cfg=has_foo
    // Can then be used as `#[cfg(has_foo)]` when emitted

    // Emit custom env data:
    //     cargo:rustc-env=foo=bar
    // Can then be fetched with `env!("foo")`

    // LLVM
    let outdir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cwd = env::current_dir().unwrap();
    let llvm_prefix_env = env::var(ENV_LLVM_PREFIX).expect(ENV_LLVM_PREFIX);

    if let Err(_) = which::which("lumen-tblgen") {
        fail(
            "Unable to locate lumen-tblgen!\n\
             It is required for the build, make sure it is built first and try again.",
        );
    }

    let project_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmakelists_path = project_path
        .parent()
        .map(|p| p.join("codegen_llvm"))
        .unwrap();
    let dialect_eir_path = cmakelists_path.join("lib").join("lumen").join("EIR");
    let ir_path = dialect_eir_path.join("IR");
    let tablegen_input_path = ir_path.join("EIRBase.td");

    println!("cargo:rerun-if-changed={}", tablegen_input_path.display());

    let term_encoding_rs_dest = outdir.join("term_encoding.rs");
    let flags = vec![
        "-gen-rust-eir-encoding-defs".to_owned(),
        "--write-if-changed".to_owned(),
        format!("-I={}", ir_path.to_str().unwrap()),
        format!("-I={}/include", llvm_prefix_env.as_str()),
        format!("-o={}", term_encoding_rs_dest.to_str().unwrap()),
        tablegen_input_path.to_str().unwrap().to_owned(),
    ];

    let include_dir = outdir.join("include");
    fs::create_dir_all(include_dir.join("lumen/term")).unwrap();
    fs::copy(
        cwd.join("c_src/include/lumen/term/Encoding.h"),
        include_dir.join("lumen/term/Encoding.h"),
    )
    .unwrap();

    println!("cargo:include={}", include_dir.display());
    println!("cargo:output_dir={}", outdir.display());

    tblgen(flags.as_slice());
}

fn tblgen(args: &[String]) {
    let result = Command::new("lumen-tblgen").args(args).output();
    match result {
        Ok(output) => {
            if output.status.success() {
                return;
            }
            String::from_utf8(output.stdout)
                .map(|s| println!("{}", s.trim_end()))
                .unwrap();
            String::from_utf8(output.stderr)
                .map(|s| println!("{}", s.trim_end()))
                .unwrap();
        }
        Err(e) => fail(&format!("command failed: {}", e)),
    }
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}

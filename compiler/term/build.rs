extern crate cmake;
extern crate which;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const ENV_LLVM_PREFIX: &'static str = "LLVM_SYS_90_PREFIX";

fn main() {
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "wasm32" {
        return;
    }

    // Emit custom cfg types:
    //     cargo:rustc-cfg=has_foo
    // Can then be used as `#[cfg(has_foo)]` when emitted

    // Emit custom env data:
    //     cargo:rustc-env=foo=bar
    // Can then be fetched with `env!("foo")`

    // LLVM
    let llvm_prefix_env = env::var(ENV_LLVM_PREFIX).expect(ENV_LLVM_PREFIX);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);

    if let Err(_) = which::which("cmake") {
        fail(
            "Unable to locate CMake!\n\
             It is required for the build, make sure you have a recent version installed.",
        );
    }

    let mut use_ninja = true;
    if let Err(_) = which::which("ninja") {
        use_ninja = false;
        warn(
            "Unable to locate Ninja, your CMake builds may take unncessarily long.\n\
             It is highly recommended that you install Ninja.",
        );
    }
    println!("cargo:rerun-if-changed=use_ninja={}", use_ninja);

    let project_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmakelists_path = project_path
        .parent()
        .map(|p| p.join("codegen/lib"))
        .unwrap();
    let mut config = &mut cmake::Config::new(&cmakelists_path);
    if use_ninja {
        config = config.generator("Ninja");
    }
    let build_shared = if env::var_os("LLVM_BUILD_STATIC").is_some() {
        "OFF"
    } else {
        "ON"
    };
    let outdir = config
        .define("LUMEN_BUILD_COMPILER", "ON")
        .define("LUMEN_BUILD_TESTS", "OFF")
        .define("BUILD_SHARED_LIBS", build_shared)
        .define("LLVM_PREFIX", llvm_prefix_env.as_str())
        .env("LLVM_PREFIX", llvm_prefix_env.as_str())
        .always_configure(true)
        .build_target("CopyRustEncodingGenOutput")
        .very_verbose(false)
        .build();

    let compiler_path = cmakelists_path.join("lumen").join("compiler");
    let dialect_eir_path = compiler_path.join("Dialect").join("EIR");
    let target_lib_path = compiler_path.join("Target");
    let encoding_gen_lib_path = dialect_eir_path.join("Tools");
    rerun_if_changed_anything_in_dir(&target_lib_path);
    rerun_if_changed_anything_in_dir(&encoding_gen_lib_path);

    let term_encoding_rs_src = outdir.join("build/lumen/compiler").join("term_encoding.rs");
    let term_encoding_rs_dest = outdir.join("term_encoding.rs");

    fs::copy(term_encoding_rs_src, term_encoding_rs_dest)
        .expect("unable to copy term_encoding.rs!");
}

pub fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| !ignore_changes(Path::new(&*e.file_name())))
        .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn ignore_changes(name: &Path) -> bool {
    return name
        .file_name()
        .map(|f| {
            let name = f.to_string_lossy();
            if name.starts_with(".") {
                return true;
            }

            if name.ends_with(".cpp") || name.ends_with(".h") || name.ends_with(".td") {
                return false;
            }

            true
        })
        .unwrap_or(false);
}

fn warn(s: &str) {
    println!("cargo:warning={}", s);
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}

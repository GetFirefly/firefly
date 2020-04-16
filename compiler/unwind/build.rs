use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = env::var("TARGET").expect("TARGET was not set");

    if cfg!(feature = "llvm-libunwind")
        && ((target.contains("linux") && !target.contains("musl")) || target.contains("fuchsia"))
    {
        // Build the unwinding from libunwind C/C++ source code.
        llvm_libunwind::compile();
    } else if target.contains("linux") {
        if target.contains("musl") {
            // linking for musl is handled in lib.rs
            llvm_libunwind::compile();
        } else if !target.contains("android") {
            println!("cargo:rustc-link-lib=gcc_s");
        }
    } else if target.contains("freebsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("rumprun") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("netbsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("openbsd") {
        if target.contains("sparc64") {
            println!("cargo:rustc-link-lib=gcc");
        } else {
            println!("cargo:rustc-link-lib=c++abi");
        }
    } else if target.contains("solaris") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("dragonfly") {
        println!("cargo:rustc-link-lib=gcc_pic");
    } else if target.contains("pc-windows-gnu") {
        // This is handled in the target spec with late_link_args_[static|dynamic]

        // cfg!(bootstrap) doesn't work in build scripts
        if env::var("RUSTC_STAGE").ok() == Some("0".to_string()) {
            println!("cargo:rustc-link-lib=static-nobundle=gcc_eh");
            println!("cargo:rustc-link-lib=static-nobundle=pthread");
        }
    } else if target.contains("uwp-windows-gnu") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("fuchsia") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("haiku") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("redox") {
        // redox is handled in lib.rs
    } else if target.contains("cloudabi") {
        println!("cargo:rustc-link-lib=unwind");
    }
}

mod llvm_libunwind {
    use std::env;

    /// Compile the libunwind C/C++ source code.
    pub fn compile() {
        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap();
        let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap();
        let target_endian_little = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap() != "big";
        let cfg = &mut cc::Build::new();

        cfg.cpp(true);
        cfg.cpp_set_stdlib(None);
        cfg.warnings(false);

        // libunwind expects a __LITTLE_ENDIAN__ macro to be set for LE archs, cf. #65765
        if target_endian_little {
            cfg.define("__LITTLE_ENDIAN__", Some("1"));
        }

        if target_env == "msvc" {
            // Don't pull in extra libraries on MSVC
            cfg.flag("/Zl");
            cfg.flag("/EHsc");
            cfg.define("_CRT_SECURE_NO_WARNINGS", None);
            cfg.define("_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS", None);
        } else {
            cfg.flag("-std=c99");
            cfg.flag("-std=c++11");
            cfg.flag("-nostdinc++");
            cfg.flag("-fno-exceptions");
            cfg.flag("-fno-rtti");
            cfg.flag("-fstrict-aliasing");
            cfg.flag("-funwind-tables");
        }

        let mut unwind_sources = vec![
            "Unwind-EHABI.cpp",
            "Unwind-seh.cpp",
            "Unwind-sjlj.c",
            "UnwindLevel1-gcc-ext.c",
            "UnwindLevel1.c",
            "UnwindRegistersRestore.S",
            "UnwindRegistersSave.S",
            "libunwind.cpp",
        ];

        if target_vendor == "apple" {
            unwind_sources.push("Unwind_AppleExtras.cpp");
        }

        let cwd = env::current_dir().unwrap();
        cfg.include(cwd.join("c_src/include"));
        for src in unwind_sources {
            cfg.file(cwd.join("c_src").join(src));
        }

        if target_env == "musl" {
            // use the same C compiler command to compile C++ code so we do not need to setup the
            // C++ compiler env variables on the builders
            cfg.cpp(false);
            // linking for musl is handled in lib.rs
            cfg.cargo_metadata(false);
            println!(
                "cargo:rustc-link-search=native={}",
                env::var("OUT_DIR").unwrap()
            );
        }

        cfg.compile("unwind");
    }
}

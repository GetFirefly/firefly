use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = env::var("TARGET").expect("TARGET was not set");

    llvm_libunwind::compile();

    if target.contains("android") {
        let build = cc::Build::new();

        // Since ndk r23 beta 3 `libgcc` was replaced with `libunwind` thus
        // check if we have `libunwind` available and if so use it. Otherwise
        // fall back to `libgcc` to support older ndk versions.
        let has_unwind = build
            .is_flag_supported("-lunwind")
            .expect("Unable to invoke compiler");

        if has_unwind {
            println!("cargo:rustc-cfg=feature=\"system-llvm-libunwind\"");
        }
    }
}

mod llvm_libunwind {
    use std::env;
    use std::ffi::OsStr;
    use std::fs;

    /// Compile the libunwind C/C++ source code.
    pub fn compile() {
        let target = env::var("TARGET").expect("TARGET was not set");
        let cwd = env::current_dir().unwrap();
        let out_dir = env::var("OUT_DIR").unwrap();

        let mut cc_cfg = cc::Build::new();
        let mut cpp_cfg = cc::Build::new();

        cpp_cfg.cpp(true);
        cpp_cfg.cpp_set_stdlib(None);
        cpp_cfg.flag("-nostdinc++");
        cpp_cfg.flag("-fno-exceptions");
        cpp_cfg.flag("-fno-rtti");
        cpp_cfg.flag_if_supported("-fvisibility-global-new-delete-hidden");

        for cfg in [&mut cc_cfg, &mut cpp_cfg].iter_mut() {
            cfg.warnings(false);
            cfg.debug(false);
            cfg.opt_level(3);
            cfg.flag("-fstrict-aliasing");
            cfg.flag_if_supported("-fvisibility-hidden");
            cfg.define("_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS", None);
            cfg.include(cwd.join("c_src/include"));
            cfg.cargo_metadata(false);

            if target.contains("x86_64-fortanix-unknown-sgx") {
                cfg.static_flag(true);
                cfg.flag("-fno-stack-protector");
                cfg.flag("-ffreestanding");
                cfg.flag("-fexceptions");

                // easiest way to undefine since no API available in cc::Build to undefine
                cfg.flag("-U_FORTIFY_SOURCE");
                cfg.define("_FORTIFY_SOURCE", "0");
                cfg.define("RUST_SGX", "1");
                cfg.define("__NO_STRING_INLINES", None);
                cfg.define("__NO_MATH_INLINES", None);
                cfg.define("_LIBUNWIND_IS_BAREMETAL", None);
                cfg.define("__LIBUNWIND_IS_NATIVE_ONLY", None);
                cfg.define("NDEBUG", None);
            }
            if target.contains("windows") {
                cfg.define("_LIBUNWIND_HIDE_SYMBOLS", "1");
                cfg.define("_LIBUNWIND_IS_NATIVE_ONLY", "1");
            }
        }

        // Don't set this for clang
        // By default, Clang builds C code in GNU C17 mode.
        // By default, Clang builds C++ code according to the C++98 standard,
        // with many C++11 features accepted as extensions.
        if cc_cfg.get_compiler().is_like_gnu() {
            cc_cfg.flag("-std=c99");
        }
        if cpp_cfg.get_compiler().is_like_gnu() {
            cpp_cfg.flag("-std=c++11");
        }

        if target.contains("x86_64-fortanix-unknown-sgx") || target.contains("musl") {
            // use the same GCC C compiler command to compile C++ code so we do not need to setup the
            // C++ compiler env variables on the builders.
            // Don't set this for clang++, as clang++ is able to compile this without libc++.
            if cpp_cfg.get_compiler().is_like_gnu() {
                cpp_cfg.cpp(false);
                cpp_cfg.compiler(cc_cfg.get_compiler().path());
            }
        }

        let c_sources = vec![
            "Unwind-sjlj.c",
            "UnwindLevel1-gcc-ext.c",
            "UnwindLevel1.c",
            "UnwindRegistersRestore.S",
            "UnwindRegistersSave.S",
        ];

        let cpp_sources = vec!["Unwind-EHABI.cpp", "Unwind-seh.cpp", "libunwind.cpp"];
        let cpp_len = cpp_sources.len();

        for src in c_sources.iter() {
            cc_cfg.file(cwd.join("c_src").join(src).canonicalize().unwrap());
        }
        for src in cpp_sources.iter() {
            cpp_cfg.file(cwd.join("c_src").join(src).canonicalize().unwrap());
        }

        cpp_cfg.compile("unwind-cpp");

        // FIXME: https://github.com/alexcrichton/cc-rs/issues/545#issuecomment-679242845
        let mut count = 0;
        for entry in fs::read_dir(&out_dir).unwrap() {
            let file = entry.unwrap().path().canonicalize().unwrap();
            if file.is_file() && file.extension() == Some(OsStr::new("o")) {
                // file name starts with "<unique-prefix>-Unwind-EHABI", "<unique-prefix>-Unwind-seh" or "<unique-prefix>-libunwind"
                let (_prefix, file_name) = file
                    .file_name()
                    .unwrap()
                    .to_str()
                    .expect("UTF-8 file name")
                    .split_once('-')
                    .unwrap();
                if cpp_sources
                    .iter()
                    .any(|f| file_name.starts_with(&f[..f.len() - 4]))
                {
                    cc_cfg.object(&file);
                    count += 1;
                }
            }
        }
        assert_eq!(cpp_len, count, "Can't get object files from {:?}", &out_dir);

        cc_cfg.compile("unwind");

        println!("cargo:rustc-link-search=native={}", out_dir);
    }
}

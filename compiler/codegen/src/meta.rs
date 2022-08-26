use std::borrow::Borrow;
use std::path::{Path, PathBuf};

use firefly_intern::Symbol;
use firefly_llvm as llvm;
use firefly_session::Options;
use firefly_target::spec::PanicStrategy;
use firefly_util::fs::NativeLibraryKind;

use crate::linker;

/// Represents the results of compiling a single module
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledModule {
    pub name: Symbol,
    pub object: Option<PathBuf>,
    pub dwarf_object: Option<PathBuf>,
    pub bytecode: Option<PathBuf>,
}
impl CompiledModule {
    pub fn object(&self) -> Option<&Path> {
        self.object.as_deref()
    }

    pub fn bytecode(&self) -> Option<&Path> {
        self.bytecode.as_deref()
    }
}

/// Represents the result of compiling a single application
#[derive(Debug)]
pub struct CodegenResults {
    pub app_name: Symbol,
    pub modules: Vec<CompiledModule>,
    pub project_info: ProjectInfo,
}

#[derive(Debug)]
pub struct Dependency {
    pub name: Symbol,
    pub source: Option<PathBuf>,
}

#[derive(Debug)]
pub struct ProjectInfo {
    pub target_cpu: String,
    pub exported_symbols: Vec<String>,
    pub native_libraries: Vec<NativeLibrary>,
    pub used_libraries: Vec<NativeLibrary>,
    pub used_deps: Vec<Dependency>,
    pub windows_subsystem: Option<String>,
}
impl ProjectInfo {
    pub fn new(options: &Options) -> Self {
        let mut info = Self::default();
        info.target_cpu = llvm::target::target_cpu(options).to_string();
        info.exported_symbols = linker::exported_symbols(options);

        // We always add dependencies on our core runtime crates
        let fireflylib_dir = options
            .target_tlib_path
            .as_ref()
            .map(|t| t.dir.clone())
            .unwrap_or_else(|| options.host_tlib_path.dir.clone());
        let prefix = &options.target.options.staticlib_prefix;
        info.used_deps
            .push(match options.target.options.panic_strategy {
                PanicStrategy::Abort => Dependency {
                    name: Symbol::intern("panic_abort"),
                    source: Some(fireflylib_dir.join(&format!("{}panic_abort.rlib", prefix))),
                },
                PanicStrategy::Unwind => Dependency {
                    name: Symbol::intern("panic_unwind"),
                    source: Some(fireflylib_dir.join(&format!("{}panic_unwind.rlib", prefix))),
                },
            });
        if options.target.options.is_like_wasm {
            info.used_libraries.push(NativeLibrary {
                kind: NativeLibraryKind::Static {
                    bundle: None,
                    whole_archive: None,
                },
                name: Some("firefly_web".to_string()),
                verbatim: None,
            });
        } else {
            info.used_deps.push(Dependency {
                name: Symbol::intern("panic"),
                source: Some(fireflylib_dir.join(&format!("{}panic.rlib", prefix))),
            });
            info.used_deps.push(Dependency {
                name: Symbol::intern("unwind"),
                source: Some(fireflylib_dir.join(&format!("{}unwind.rlib", prefix))),
            });
            info.used_libraries.push(NativeLibrary {
                kind: NativeLibraryKind::Static {
                    bundle: None,
                    whole_archive: Some(true),
                },
                name: Some("firefly_rt_tiny".to_string()),
                verbatim: None,
            });
            /*
            info.used_deps.push(Dependency {
                name: Symbol::intern("firefly_otp"),
                source: Some(fireflylib_dir.join(&format!("{}firefly_otp.rlib", prefix))),
            });
            */
        }

        // Add user-provided libraries
        for (name, _, kind) in options.link_libraries.iter() {
            info.native_libraries.push(NativeLibrary {
                kind: *kind,
                name: Some(name.clone()),
                verbatim: None,
            });
        }

        // Add platform-specific libraries that must be linked to
        let mut platform_libs = vec![("c", NativeLibraryKind::Unspecified)];
        let target_os: &str = options.target.options.os.borrow();
        let target_env: &str = options.target.options.env.borrow();
        if target_os == "android" {
            platform_libs.extend_from_slice(&[
                ("dl", NativeLibraryKind::Unspecified),
                ("log", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "freebsd" {
            platform_libs.extend_from_slice(&[
                ("execinfo", NativeLibraryKind::Unspecified),
                ("pthread", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "netbsd" {
            platform_libs.extend_from_slice(&[
                ("pthread", NativeLibraryKind::Unspecified),
                ("rt", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "dragonfly" || target_os == "openbsd" {
            platform_libs.extend_from_slice(&[("pthread", NativeLibraryKind::Unspecified)]);
        } else if target_os == "solaris" {
            platform_libs.extend_from_slice(&[
                ("socket", NativeLibraryKind::Unspecified),
                ("posix4", NativeLibraryKind::Unspecified),
                ("pthread", NativeLibraryKind::Unspecified),
                ("resolv", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "illumos" {
            platform_libs.extend_from_slice(&[
                ("socket", NativeLibraryKind::Unspecified),
                ("posix4", NativeLibraryKind::Unspecified),
                ("pthread", NativeLibraryKind::Unspecified),
                ("resolv", NativeLibraryKind::Unspecified),
                ("nsl", NativeLibraryKind::Unspecified),
                // Use libumem for the (malloc-compatible) allocator
                ("umem", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "macos" {
            platform_libs.extend_from_slice(&[
                ("System", NativeLibraryKind::Unspecified),
                // res_init and friends require -lresolv on macOS/iOS.
                // See #41582 and https://blog.achernya.com/2013/03/os-x-has-silly-libsystem.html
                ("resolv", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "ios" {
            platform_libs.extend_from_slice(&[
                ("System", NativeLibraryKind::Unspecified),
                ("objc", NativeLibraryKind::Unspecified),
                ("Security", NativeLibraryKind::Framework { as_needed: None }),
                (
                    "Foundation",
                    NativeLibraryKind::Framework { as_needed: None },
                ),
                ("resolv", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "fuchsia" {
            platform_libs.extend_from_slice(&[
                ("zircon", NativeLibraryKind::Unspecified),
                ("fdio", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "linux" && target_env == "uclibc" {
            platform_libs.extend_from_slice(&[("dl", NativeLibraryKind::Unspecified)]);
        }

        for (name, kind) in platform_libs.drain(..) {
            info.native_libraries.push(NativeLibrary {
                kind,
                name: Some(name.to_owned()),
                verbatim: None,
            });
        }

        info
    }
}
impl Default for ProjectInfo {
    fn default() -> Self {
        Self {
            target_cpu: llvm::target::host_cpu().to_string(),
            exported_symbols: vec![],
            native_libraries: vec![],
            used_libraries: vec![],
            used_deps: vec![],
            windows_subsystem: None,
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct NativeLibrary {
    pub kind: NativeLibraryKind,
    pub name: Option<String>,
    pub verbatim: Option<bool>,
}

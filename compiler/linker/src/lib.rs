#![feature(once_cell)]
#![feature(try_blocks)]
#![feature(associated_type_bounds)]

mod archive;
mod command;
mod link;
mod linker;
mod rpath;

use std::borrow::Borrow;
use std::ffi::OsStr;
use std::mem;
use std::path::{Path, PathBuf};

use firefly_intern::Symbol;
use firefly_llvm as llvm;
use firefly_session::{Options, ProjectType, Strip};
use firefly_target::spec::{LinkOutputKind, PanicStrategy};
use firefly_util::fs::NativeLibraryKind;

pub(crate) use self::command::Command;

pub use self::link::link_binary;

/// Linker abstraction used by `link_binary` to build up the command to invoke a linker.
///
/// This trait is the total list of requirements needed the linker backend and
/// represents the meaning of each option being passed down.
///
/// This trait is then used to dispatch on whether a GNU-like linker (generally `ld.exe`) or an
/// MSVC linker (e.g., `link.exe`) is being used. See the individual implementations in `crate::linker`.
pub trait Linker {
    fn cmd(&mut self) -> &mut Command;
    fn is_ld(&self) -> bool {
        false
    }
    fn set_output_kind(&mut self, output_kind: LinkOutputKind, out_filename: &Path);
    fn link_dylib(&mut self, lib: &str, verbatim: bool, as_needed: bool);
    fn link_rust_dylib(&mut self, lib: &str, path: &Path);
    fn link_framework(&mut self, framework: &str, as_needed: bool);
    fn link_staticlib(&mut self, lib: &str, verbatim: bool);
    fn link_rlib(&mut self, lib: &Path);
    fn link_whole_rlib(&mut self, lib: &Path);
    fn link_whole_staticlib(&mut self, lib: &str, verbatim: bool, search_path: &[PathBuf]);
    fn include_path(&mut self, path: &Path);
    fn framework_path(&mut self, path: &Path);
    fn output_filename(&mut self, path: &Path);
    fn add_object(&mut self, path: &Path);
    fn gc_sections(&mut self, keep_metadata: bool);
    fn no_gc_sections(&mut self);
    fn full_relro(&mut self);
    fn partial_relro(&mut self);
    fn no_relro(&mut self);
    fn optimize(&mut self);
    fn pgo_gen(&mut self);
    fn control_flow_guard(&mut self);
    fn debuginfo(&mut self, strip: Strip);
    fn no_crt_objects(&mut self);
    fn no_default_libraries(&mut self);
    fn export_symbols(&mut self, tmpdir: &Path, project_type: ProjectType, symbols: &[String]);
    fn exported_symbol_means_used_symbol(&self) -> bool {
        true
    }
    fn subsystem(&mut self, subsystem: &str);
    fn group_start(&mut self);
    fn group_end(&mut self);
    fn linker_plugin_lto(&mut self);
    fn add_eh_frame_header(&mut self) {}
    fn add_no_exec(&mut self) {}
    fn add_as_needed(&mut self) {}
    fn reset_per_library_state(&mut self) {}
}
impl dyn Linker + '_ {
    pub fn arg(&mut self, arg: impl AsRef<OsStr>) {
        self.cmd().arg(arg);
    }

    pub fn args(&mut self, args: impl IntoIterator<Item: AsRef<OsStr>>) {
        self.cmd().args(args);
    }

    pub fn take_cmd(&mut self) -> Command {
        mem::replace(self.cmd(), Command::new(""))
    }
}

/// Represents the artifacts produced by compiling a single application
#[derive(Debug)]
pub struct AppArtifacts {
    pub name: Symbol,
    pub modules: Vec<ModuleArtifacts>,
    pub project_info: ProjectInfo,
}

/// Represents the artifacts produced by compiling a single module
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleArtifacts {
    pub name: Symbol,
    pub object: Option<PathBuf>,
    pub dwarf_object: Option<PathBuf>,
    pub bytecode: Option<PathBuf>,
}
impl ModuleArtifacts {
    pub fn object(&self) -> Option<&Path> {
        self.object.as_deref()
    }

    pub fn bytecode(&self) -> Option<&Path> {
        self.bytecode.as_deref()
    }
}

#[derive(Debug, Clone, Hash)]
pub struct Dependency {
    pub name: Symbol,
    pub source: Option<PathBuf>,
}

#[derive(Clone, Debug, Hash)]
pub struct NativeLibrary {
    pub kind: NativeLibraryKind,
    pub name: Option<String>,
    pub verbatim: Option<bool>,
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
        info.exported_symbols = options.exported_symbols();

        // We always add dependencies on our core runtime crates
        let fireflylib_dir = options
            .target_tlib_path
            .as_ref()
            .map(|t| t.dir.clone())
            .unwrap_or_else(|| options.host_tlib_path.dir.clone());
        let prefix = &options.target.options.staticlib_prefix;
        match options.target.options.panic_strategy {
            PanicStrategy::Abort => {
                info.used_deps.push(Dependency {
                    name: Symbol::intern("panic_abort"),
                    source: Some(fireflylib_dir.join(&format!("{}panic_abort.rlib", prefix))),
                });
            }
            PanicStrategy::Unwind => {
                info.used_deps.push(Dependency {
                    name: Symbol::intern("panic_unwind"),
                    source: Some(fireflylib_dir.join(&format!("{}panic_unwind.rlib", prefix))),
                });
                info.used_deps.push(Dependency {
                    name: Symbol::intern("unwind"),
                    source: Some(fireflylib_dir.join(&format!("{}unwind.rlib", prefix))),
                });
            }
        }
        if options.target.options.is_like_wasm {
            info.used_deps.push(Dependency {
                name: Symbol::intern("firefly_emulator"),
                source: Some(fireflylib_dir.join(&format!("{}firefly_emulator.rlib", prefix))),
            });
            /*
            info.used_libraries.push(NativeLibrary {
                kind: NativeLibraryKind::Static {
                    bundle: None,
                    whole_archive: None,
                },
                name: Some("firefly_web".to_string()),
                verbatim: None,
            });
            */
        } else {
            info.used_deps.push(Dependency {
                name: Symbol::intern("firefly_emulator"),
                source: Some(fireflylib_dir.join(&format!("{}firefly_emulator.a", prefix))),
            });
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
                ("gcc_s", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "netbsd" {
            platform_libs.extend_from_slice(&[
                ("pthread", NativeLibraryKind::Unspecified),
                ("rt", NativeLibraryKind::Unspecified),
                ("gcc_s", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "dragonfly" {
            platform_libs.extend_from_slice(&[
                ("pthread", NativeLibraryKind::Unspecified),
                ("gcc_pic", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "openbsd" {
            platform_libs.extend_from_slice(&[
                ("pthread", NativeLibraryKind::Unspecified),
                ("c++abi", NativeLibraryKind::Unspecified),
            ]);
        } else if target_os == "solaris" {
            platform_libs.extend_from_slice(&[
                ("socket", NativeLibraryKind::Unspecified),
                ("posix4", NativeLibraryKind::Unspecified),
                ("pthread", NativeLibraryKind::Unspecified),
                ("resolv", NativeLibraryKind::Unspecified),
                ("gcc_s", NativeLibraryKind::Unspecified),
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
                ("gcc_s", NativeLibraryKind::Unspecified),
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
            if !options.crt_static(Some(options.project_type)) {
                platform_libs.extend_from_slice(&[("gcc_s", NativeLibraryKind::Unspecified)]);
            }
        } else if target_os == "linux" && target_env == "gnu" {
            platform_libs.extend_from_slice(&[
                ("m", NativeLibraryKind::Unspecified),
                ("dl", NativeLibraryKind::Unspecified),
                ("pthread", NativeLibraryKind::Unspecified),
            ]);
            if !options.crt_static(Some(options.project_type)) {
                platform_libs.extend_from_slice(&[("gcc_s", NativeLibraryKind::Unspecified)]);
            }
        } else if target_os == "linux" && target_env == "musl" {
            if !options.crt_static(Some(options.project_type)) {
                platform_libs.extend_from_slice(&[("gcc_s", NativeLibraryKind::Unspecified)]);
            } else {
                platform_libs.extend_from_slice(&[("unwind", NativeLibraryKind::Unspecified)]);
            }
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

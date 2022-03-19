use std::path::{Path, PathBuf};
use std::sync::Arc;

use liblumen_intern::Symbol;
use liblumen_session::{Options, PathKind};
use liblumen_util::fs::NativeLibraryKind;

use crate::linker::LinkerInfo;

/// Represents the results of compiling a single module
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledModule {
    name: Symbol,
    object: Option<PathBuf>,
    bytecode: Option<PathBuf>,
    bytecode_compressed: Option<PathBuf>,
}
impl CompiledModule {
    pub fn new(name: Symbol, object: Option<PathBuf>, bytecode: Option<PathBuf>) -> Self {
        Self {
            name,
            object,
            bytecode,
            bytecode_compressed: None,
        }
    }

    pub fn name(&self) -> Symbol {
        self.name
    }

    pub fn object(&self) -> Option<&Path> {
        self.object.as_deref()
    }

    pub fn bytecode(&self) -> Option<&Path> {
        self.bytecode.as_deref()
    }

    pub fn bytecode_compressed(&self) -> Option<&Path> {
        self.bytecode_compressed.as_deref()
    }
}

/// Represents the result of compiling a single application
#[derive(Debug)]
pub struct CodegenResults {
    pub app_name: Symbol,
    pub modules: Vec<CompiledModule>,
    pub windows_subsystem: Option<String>,
    pub linker_info: LinkerInfo,
    pub project_info: ProjectInfo,
}

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum Linkage {
    NotLinked,
    IncludedFromDylib,
    Static,
    Dynamic,
}

pub type DependencyList = Vec<Linkage>;

pub struct DependencySource {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
}
impl DependencySource {
    pub fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.dylib.iter().chain(self.rlib.iter()).map(|p| &p.0)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DepKind {
    Implicit,
    Explicit,
}

#[derive(Clone, Debug)]
pub enum LibSource {
    Some(PathBuf),
    None,
}
impl LibSource {
    pub fn is_some(&self) -> bool {
        if let LibSource::Some(_) = *self {
            true
        } else {
            false
        }
    }

    pub fn option(&self) -> Option<PathBuf> {
        match *self {
            LibSource::Some(ref p) => Some(p.clone()),
            LibSource::None => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LinkagePreference {
    RequireDynamic,
    RequireStatic,
}

#[derive(Debug)]
pub struct ProjectInfo {
    pub native_libraries: Vec<NativeLibrary>,
    pub used_libraries: Arc<Vec<NativeLibrary>>,
    pub link_args: Vec<String>,
    pub used_deps_static: Vec<(String, LibSource)>,
    pub used_deps_dynamic: Vec<(String, LibSource)>,
}
impl ProjectInfo {
    pub fn new(options: &Options) -> Self {
        let mut info = Self::default();
        // All other libraries are user-provided
        for (name, _, kind) in options.link_libraries.iter() {
            info.native_libraries.push(NativeLibrary {
                kind: kind.unwrap_or(NativeLibraryKind::NativeUnknown),
                name: Some(name.clone()),
                wasm_import_module: None,
            });
        }

        // Add platform-specific libraries that must be linked to
        let mut platform_libs = vec![];
        let triple = &options.target.llvm_target;
        let target_os = &options.target.target_os;
        if triple.contains("linux") {
            if triple.contains("android") {
                platform_libs = vec![
                    ("c", NativeLibraryKind::NativeUnknown),
                    ("m", NativeLibraryKind::NativeUnknown),
                    ("dl", NativeLibraryKind::NativeUnknown),
                    ("log", NativeLibraryKind::NativeUnknown),
                    ("gcc", NativeLibraryKind::NativeUnknown),
                ];
            } else if !triple.contains("musl") {
                platform_libs = vec![
                    ("c", NativeLibraryKind::NativeUnknown),
                    ("m", NativeLibraryKind::NativeUnknown),
                    ("dl", NativeLibraryKind::NativeUnknown),
                    ("rt", NativeLibraryKind::NativeUnknown),
                    ("pthread", NativeLibraryKind::NativeUnknown),
                ];
            }
        } else if target_os == "freebsd" {
            platform_libs = vec![
                ("execinfo", NativeLibraryKind::NativeUnknown),
                ("pthread", NativeLibraryKind::NativeUnknown),
            ];
        } else if target_os == "netbsd" {
            platform_libs = vec![
                ("pthread", NativeLibraryKind::NativeUnknown),
                ("rt", NativeLibraryKind::NativeUnknown),
            ];
        } else if target_os == "dragonfly" || target_os == "openbsd" {
            platform_libs = vec![("pthread", NativeLibraryKind::NativeUnknown)];
        } else if target_os == "solaris" {
            platform_libs = vec![
                ("socket", NativeLibraryKind::NativeUnknown),
                ("posix4", NativeLibraryKind::NativeUnknown),
                ("pthread", NativeLibraryKind::NativeUnknown),
                ("resolv", NativeLibraryKind::NativeUnknown),
            ];
        } else if target_os == "macos" {
            // res_init and friends require -lresolv on macOS/iOS.
            // See #41582 and http://blog.achernya.com/2013/03/os-x-has-silly-libsystem.html
            platform_libs = vec![
                ("System", NativeLibraryKind::NativeUnknown),
                ("resolv", NativeLibraryKind::NativeUnknown),
            ];
        } else if target_os == "ios" {
            platform_libs = vec![
                ("System", NativeLibraryKind::NativeUnknown),
                ("objc", NativeLibraryKind::NativeUnknown),
                ("Security", NativeLibraryKind::NativeFramework),
                ("Foundation", NativeLibraryKind::NativeFramework),
                ("resolv", NativeLibraryKind::NativeUnknown),
            ];
        } else if triple.contains("uwp") {
            // For BCryptGenRandom
            platform_libs = vec![
                ("ws2_32", NativeLibraryKind::NativeUnknown),
                ("bcrypt", NativeLibraryKind::NativeUnknown),
            ];
        } else if target_os == "windows" {
            platform_libs = vec![
                ("advapi32", NativeLibraryKind::NativeUnknown),
                ("ws2_32", NativeLibraryKind::NativeUnknown),
                ("userenv", NativeLibraryKind::NativeUnknown),
            ];
        } else if target_os == "fuchsia" {
            platform_libs = vec![
                ("zircon", NativeLibraryKind::NativeUnknown),
                ("fdio", NativeLibraryKind::NativeUnknown),
            ];
        } else if triple.contains("cloudabi") {
            platform_libs = vec![
                ("unwind", NativeLibraryKind::NativeUnknown),
                ("c", NativeLibraryKind::NativeUnknown),
                ("compiler_rt", NativeLibraryKind::NativeUnknown),
            ];
        }

        for (name, kind) in platform_libs.drain(..) {
            info.native_libraries.push(NativeLibrary {
                kind,
                name: Some(name.to_owned()),
                wasm_import_module: None,
            });
        }

        info
    }
}
impl Default for ProjectInfo {
    fn default() -> Self {
        Self {
            native_libraries: Vec::new(),
            used_libraries: Arc::new(Vec::new()),
            link_args: Vec::new(),
            used_deps_static: Vec::new(),
            used_deps_dynamic: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct NativeLibrary {
    pub kind: NativeLibraryKind,
    pub name: Option<String>,
    pub wasm_import_module: Option<String>,
}

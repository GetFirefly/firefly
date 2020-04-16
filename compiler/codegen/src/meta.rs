use std::path::{Path, PathBuf};
use std::sync::Arc;

use liblumen_session::{Options, PathKind};
use liblumen_util::fs::NativeLibraryKind;

use crate::linker::LinkerInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledModule {
    name: String,
    object: Option<PathBuf>,
    bytecode: Option<PathBuf>,
    bytecode_compressed: Option<PathBuf>,
}
impl CompiledModule {
    pub fn new(name: String, object: Option<PathBuf>, bytecode: Option<PathBuf>) -> Self {
        Self {
            name,
            object,
            bytecode,
            bytecode_compressed: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
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

#[derive(Debug)]
pub struct CodegenResults {
    pub project_name: String,
    pub modules: Vec<Arc<CompiledModule>>,
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

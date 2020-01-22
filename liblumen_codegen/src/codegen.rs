use std::path::{Path, PathBuf};
use std::sync::Arc;

use liblumen_session::Options;
use liblumen_util::fs::NativeLibraryKind;

use crate::linker::LinkerInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledModule {
    name: String,
    object: Option<PathBuf>,
    bytecode: Option<PathBuf>,
}
impl CompiledModule {
    pub fn new(name: String, object: Option<PathBuf>, bytecode: Option<PathBuf>) -> Self {
        Self {
            name,
            object,
            bytecode,
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
}

#[derive(Debug)]
pub struct CodegenResults {
    pub project_name: String,
    pub modules: Vec<Arc<CompiledModule>>,
    pub windows_subsystem: Option<String>,
    pub linker_info: LinkerInfo,
    pub project_info: ProjectInfo,
}

#[derive(Debug)]
pub struct ProjectInfo {
    pub native_libraries: Vec<NativeLibrary>,
    pub used_libraries: Arc<Vec<NativeLibrary>>,
    pub link_args: Vec<String>,
}
impl ProjectInfo {
    pub fn new(options: &Options) -> Self {
        let mut info = Self::default();
        // Always statically link the Lumen core runtime
        info.native_libraries.push(NativeLibrary {
            kind: NativeLibraryKind::NativeStatic,
            name: Some("liblumen_crt".to_string()),
            wasm_import_module: None,
        });
        // If `-C no-std` was not set, link in the appropriate Lumen runtime crate
        if options.codegen_opts.no_std.unwrap_or(false) == false {
            if options.target.arch == "wasm32" {
                info.native_libraries.push(NativeLibrary {
                    kind: NativeLibraryKind::NativeStatic,
                    name: Some("lumen_web".to_string()),
                    wasm_import_module: None,
                });
            } else {
                info.native_libraries.push(NativeLibrary {
                    kind: NativeLibraryKind::NativeStatic,
                    name: Some("lumen_runtime".to_string()),
                    wasm_import_module: None,
                });
            }
        }
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
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct NativeLibrary {
    pub kind: NativeLibraryKind,
    pub name: Option<String>,
    pub wasm_import_module: Option<String>,
}

use std::path::PathBuf;
use std::sync::Arc;

use liblumen_session::Options;
use liblumen_util::fs::NativeLibraryKind;

use crate::linker::LinkerInfo;

#[derive(Debug)]
pub struct CompiledModule {
    pub name: String,
    pub object: Option<PathBuf>,
    pub bytecode: Option<PathBuf>,
}

#[derive(Debug)]
pub struct CodegenResults {
    pub project_name: String,
    pub modules: Vec<CompiledModule>,
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
    pub fn new(_options: &Options) -> Self {
        Self::default()
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

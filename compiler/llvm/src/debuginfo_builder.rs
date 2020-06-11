use std::ffi::CStr;
use std::path::{Path, PathBuf};

use liblumen_session::Options;

use crate::sys as llvm_sys;
use crate::{Metadata, Module};

pub type DIFlags = crate::sys::debuginfo::LLVMDIFlags;

const EMPTY_BYTES: &[u8] = &[];
const PRODUCER: &'static str = "lumen";
const RUNTIME_VERSION: ::libc::c_uint = 0;
const DWO_ID: ::libc::c_uint = 0;

pub struct DebugInfoBuilder<'m> {
    builder: LLVMDIBuilderRef,
    finalized: bool,
    optimized: bool,
    cwd: PathBuf,
}
impl<'m> DebugInfoBuilder<'m> {
    pub fn new(module: &'m Module, options: &Options) -> Self {
        use llvm_sys::debuginfo::LLVMCreateDIBuilder;

        let builder = unsafe { LLVMCreateDIBuilder(module.as_ref()) };

        let cwd = env::current_dir().unwrap();
        Self {
            builder,
            finalized: false,
            optimized: options.opt_level != OptLevel::No,
            cwd,
        }
    }

    pub fn new_strict(module: &'m Module, options: &Options) -> Self {
        use llvm_sys::debuginfo::LLVMCreateDIBuilderDisallowUnresolved;

        let builder = unsafe { LLVMCreateDIBuilderDisallowUnresolved(module.as_ref()) };

        let cwd = env::current_dir().unwrap();
        Self {
            builder,
            finalized: false,
            optimized: options.opt_level != OptLevel::No,
            cwd,
        }
    }

    pub fn create_compile_unit(&self, file: Metadata) -> Metadata {
        use llvm_sys::debuginfo::LLVMDIBuilderCreateCompileUnit;

        let flags = "";
        let split = "";

        unsafe {
            LLVMDIBuilderCreateCompileUnit(
                self.builder,
                LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageC,
                file,
                PRODUCER.as_ptr() as *const libc::c_char,
                PRODUCER.len() as libc::size_t,
                self.optimized,
                flags.as_ptr() as *const libc::c_char,
                flags.len() as libc::size_t,
                RUNTIME_VERSION,
                split.as_ptr() as *const libc::c_char,
                split.len() as libc::size_t,
                LLVMDWARFEmissionKind::LLVMDWARFEmissionKindFull,
                DWO_ID,
                /* splitDebugInlining */ true,
                /* debugInfoForProfiling */ false,
            )
        }
    }

    /// Create a file descriptor to hold debugging information for a file.
    pub fn create_file(&self, file: &Path) -> Metadata {
        use llvm_sys::debuginfo::LLVMDIBuilderCreateFile;

        let cwd = self.cwd.as_path();
        let (file, dir) = if file.is_absolute() {
            // Strip the common prefix (if it is more than just '/')
            // from current directory and filename to keep things less verbose
            if let Ok(stripped) = file.strip_prefix(cwd) {
                (stripped, Some(cwd))
            } else {
                (file, None)
            }
        };

        let filename_bytes = path_to_bytes(file);
        let filename = filename_bytes.as_ptr() as *const libc::c_char;
        let filename_len = filename_bytes.len() as libc::size_t;
        let directory_bytes = match dir {
            None => EMPTY_BYTES,
            Some(d) => path_to_bytes(d),
        };
        let directory = directory_bytes.as_ptr() as *const libc::c_char;
        let directory_len = directory_bytes.len() as libc::size_t;

        unsafe {
            LLVMDIBuilderCreateFile(
                self.builder,
                filename,
                filename_len,
                directory,
                directory_len,
            )
        }
    }

    /// Create a new descriptor for the specified subprogram.
    pub fn create_function(
        &self,
        scope: Metadata,
        name: &str,
        linkage_name: &str,
        file: Metadata,
        line: usize,
        ty: Metadata,
        is_local: bool,
        is_definition: bool,
        scope_line: usize,
        flags: DIFlags,
    ) -> Metadata {
        use llvm_sys::debuginfo::LLVMDIBuilderCreateFunction;

        unsafe {
            LLVMDIBuilderCreateFunction(
                self.builder,
                scope,
                name.as_ptr() as *const libc::c_char,
                name.len() as libc::size_t,
                linkage_name.as_ptr() as *const libc::c_char,
                linkage_name.len() as libc::size_t,
                file,
                line as libc::c_uint,
                ty,
                is_local,
                is_definition,
                scope_line as libc::c_uint,
                flags,
                self.optimized,
            )
        }
    }

    /// Create a descriptor for a lexical block with the specified parent context.
    pub fn create_lexical_block(
        &self,
        scope: Metadata,
        file: Metadata,
        line: usize,
        column: usize,
    ) -> Metadata {
        use llvm_sys::debuginfo::LLVMDIBuilderCreateLexicalBlock;

        unsafe { LLVMDIBuilderCreateLexicalBlock(self.builder, scope, file, line, column) }
    }

    /// Create a descriptor for an imported function, type, or variable.
    pub fn create_imported_declaration(
        &self,
        scope: Metadata,
        decl: Metadata,
        name: &str,
        file: Metadata,
        line: usize,
    ) -> Metadata {
        use llvm_sys::debuginfo::LLVMDIBuilderCreateImportedDeclaration;

        unsafe {
            LLVMDIBuilderCreateImportedDeclaration(
                self.builder,
                scope,
                decl,
                file,
                line as libc::c_uint,
                name.as_ptr() as *const libc::c_char,
                name.len() as libc::size_t,
            )
        }
    }

    #[inline]
    pub fn build(self) {
        self.finalize()
    }

    fn finalize(&self) {
        use llvm_sys::debuginfo::LLVMDIBuilderFinalize;

        unsafe { LLVMDIBuilderFinalize(self.builder) }
    }
}

impl<'m> Drop for DebugInfoBuilder<'m> {
    fn drop(&mut self) {
        use llvm_sys::debuginfo::LLVMDisposeDIBuilder;

        if !self.finalized {
            self.finalize()
        }

        unsafe { LLVMDisposeDIBuilder(self.builder) }
    }
}

#[cfg(unix)]
fn path_to_bytes(path: &Path) -> &[u8] {
    use std::os::unix::ffi::OsStrExt;

    path.as_os_str().as_bytes()
}

#[cfg(windows)]
fn path_to_bytes(path: &Path) -> &[u8] {
    use std::os::windows::ffi::OsStrExt;

    path.as_os_str().as_bytes()
}

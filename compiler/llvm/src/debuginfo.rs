#![allow(non_upper_case_globals)]

use std::path::{Path, PathBuf};

use liblumen_session::{OptLevel, Options};

use crate::ir::*;
use crate::support::StringRef;

const PRODUCER: &'static str = "lumen";
const RUNTIME_VERSION: u32 = 0;
const DWO_ID: u32 = 0;

extern "C" {
    type LlvmDiBuilder;
}

bitflags::bitflags! {
    /// Represents LLVM debug info flags
    #[repr(C)]
    pub struct DIFlags: u32 {
        const Zero = 0;
        const Private = 1;
        const Protected = 2;
        const Public = 3;
        const FwdDecl = 1 << 2;
        const AppleBlock = 1 << 3;
        const ReservedBit4 = 1 << 4;
        const Virtual = 1 << 5;
        const Artificial = 1 << 6;
        const Explicit = 1 << 7;
        const Prototyped = 1 << 8;
        const ObjcClassComplete = 1 << 9;
        const ObjectPointer = 1 << 10;
        const Vector = 1 << 11;
        const StaticMember = 1 << 12;
        const LValueReference = 1 << 13;
        const RValueReference = 1 << 14;
        const Reserved = 1 << 15;
        const SingleInheritance = 1 << 16;
        const MultipleInheritance = 2 << 16;
        const VirtualInheritance = 3 << 16;
        const IntroducedVirtual = 1 << 18;
        const BitField = 1 << 19;
        const NoReturn = 1 << 20;
        const TypePassByValue = 1 << 22;
        const TypePassByReference = 1 << 23;
        const EnumClass = 1 << 24;
        #[deprecated]
        const FixedEnum = Self::EnumClass.bits;
        const Thunk = 1 << 25;
        const NonTrivial = 1 << 26;
        const BigEndian = 1 << 27;
        const LittleEndian = 1 << 28;
        const IndirectVirtualBase = (1 << 2) | (1 << 5);
        const Accessibility = Self::Private.bits | Self::Protected.bits | Self::Public.bits;
        const PtrToMemberRep = Self::SingleInheritance.bits | Self::MultipleInheritance.bits | Self::VirtualInheritance.bits;
    }
}

/// The DWARF source language the debug info was generated from
///
/// This represents the set of source languages known to DWARF, not all source languages
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum DwarfSourceLanguage {
    C89 = 0,
    C,
    Ada83,
    Cpp,
    Cobol74,
    Cobol85,
    Fortran77,
    Fortran90,
    Pascal83,
    Modula2,
    Java,
    C99,
    Ada95,
    Fortran95,
    PLI,
    ObjC,
    ObjCpp,
    UPC,
    D,
    Python,
    OpenCL,
    Go,
    Modula3,
    Haskell,
    Cpp03,
    Cpp11,
    OCaml,
    Rust,
    C11,
    Swift,
    Julia,
    Dylan,
    Cpp14,
    Fortran03,
    Fortran08,
    RenderScript,
    BLISS,
    Mips_Assembler,
    GOOGLE_RenderScript,
    BORLAND_Delphi,
}

/// The amount of debug information to emit.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DwarfEmissionKind {
    None = 0,
    Full,
    LineTablesOnly,
}

/// An LLVM DWARF type encoding.
pub type DwarfTypeEncoding = u32;

/// Describes the kind of macro declaration used for LLVMDIBuilderCreateMacro
///
/// See llvm::dwarf::MacinfoRecordType
///
/// Corresponds to DW_MACINFO_* constants in the DWARF spec
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DwarfMacinfoRecordType {
    Define = 0x01,
    Macro = 0x02,
    StartFile = 0x03,
    EndFile = 0x04,
    VendorExt = 0xff,
}

/// The current debug metadata version number
pub fn metadata_version() -> u32 {
    extern "C" {
        fn LLVMDebugMetadataVersion() -> u32;
    }
    unsafe { LLVMDebugMetadataVersion() }
}

/// Represents metadata that contains location information for the associated item
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DebugLocation(Metadata);
impl DebugLocation {
    /// Returns the line number of this location
    pub fn line(self) -> u32 {
        extern "C" {
            fn LLVMDILocationGetLine(loc: DebugLocation) -> u32;
        }
        unsafe { LLVMDILocationGetLine(self) }
    }

    /// Returns the column number of this location
    pub fn column(self) -> u32 {
        extern "C" {
            fn LLVMDILocationGetColumn(loc: DebugLocation) -> u32;
        }
        unsafe { LLVMDILocationGetColumn(self) }
    }

    /// Returns the scope metadata of this location
    pub fn scope(self) -> DebugScope {
        extern "C" {
            fn LLVMDILocationGetScope(loc: DebugLocation) -> DebugScope;
        }
        unsafe { LLVMDILocationGetScope(self) }
    }
}
impl TryFrom<Metadata> for DebugLocation {
    type Error = InvalidTypeCastError;

    fn try_from(metadata: Metadata) -> Result<Self, Self::Error> {
        match metadata.kind() {
            MetadataKind::DILocation => Ok(Self(metadata)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents metadata that contains the scope for a debug location
///
/// * file
/// * type
/// * compileunit
/// * localscope
/// * subprogram
/// * lexicalblock
/// * lexicalblockfile
/// * namespace
/// * module
/// * commonblock
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DebugScope(Metadata);
impl DebugScope {
    pub fn file(self) -> DebugFile {
        extern "C" {
            fn LLVMDIScopeGetFile(scope: DebugScope) -> DebugFile;
        }
        unsafe { LLVMDIScopeGetFile(self) }
    }
}
impl TryFrom<Metadata> for DebugScope {
    type Error = InvalidTypeCastError;

    fn try_from(metadata: Metadata) -> Result<Self, Self::Error> {
        match metadata.kind() {
            MetadataKind::DIBasicType
            | MetadataKind::DIDerivedType
            | MetadataKind::DICompositeType
            | MetadataKind::DISubroutineType
            | MetadataKind::DIFile
            | MetadataKind::DICompileUnit
            | MetadataKind::DISubprogram
            | MetadataKind::DILexicalBlock
            | MetadataKind::DILexicalBlockFile
            | MetadataKind::DINamespace
            | MetadataKind::DIMacroFile
            | MetadataKind::DICommonBlock
            | MetadataKind::DIStringType => Ok(Self(metadata)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents metadata that contains information about the source file associated with a scope
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DebugFile(Metadata);
impl DebugFile {
    /// Returns the directory containing this file
    pub fn directory(self) -> StringRef {
        extern "C" {
            fn LLVMDIFileGetDirectory(file: DebugFile) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMDIFileGetDirectory(self)) }
    }

    /// Returns the name of this file
    pub fn filename(self) -> StringRef {
        extern "C" {
            fn LLVMDIFileGetFilename(file: DebugFile) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMDIFileGetFilename(self)) }
    }
}
impl TryFrom<Metadata> for DebugFile {
    type Error = InvalidTypeCastError;

    fn try_from(metadata: Metadata) -> Result<Self, Self::Error> {
        match metadata.kind() {
            MetadataKind::DIFile => Ok(Self(metadata)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// This struct represents an LLVM debug info builder for a specific module.
///
/// It holds a reference to the module it is being built for, to ensure it doesn't
/// outlive the module, and to ensure there are no mutable references alive elsewhere
/// while we're mutating the module.
pub struct DebugInfoBuilder<'m> {
    _module: &'m mut OwnedModule,
    builder: *const LlvmDiBuilder,
    finalized: bool,
    optimized: bool,
    cwd: PathBuf,
}
impl<'m> DebugInfoBuilder<'m> {
    /// Construct a builder for the given module, and collect unresolved nodes attached
    /// to the module in order to resolve cycles during a call to LLVMDIBuilderFinalize
    pub fn new(module: &'m mut OwnedModule, options: &Options) -> Self {
        extern "C" {
            fn LLVMCreateDIBuilder(module: Module) -> *const LlvmDiBuilder;
        }

        let builder = unsafe { LLVMCreateDIBuilder(module.as_mut()) };
        let cwd = std::env::current_dir().unwrap();

        Self {
            _module: module,
            builder,
            finalized: false,
            optimized: options.opt_level != OptLevel::No,
            cwd,
        }
    }

    /// Construct a builder for the given module, and do not allow for unresolved nodes attached to the module
    pub fn new_strict(module: &'m mut OwnedModule, options: &Options) -> Self {
        extern "C" {
            fn LLVMCreateDIBuilderDisallowUnresolved(module: Module) -> *const LlvmDiBuilder;
        }

        let builder = unsafe { LLVMCreateDIBuilderDisallowUnresolved(module.as_mut()) };
        let cwd = std::env::current_dir().unwrap();

        Self {
            _module: module,
            builder,
            finalized: false,
            optimized: options.opt_level != OptLevel::No,
            cwd,
        }
    }

    pub fn create_compile_unit<S1, S2, S3>(&self, file: DebugFile) -> Metadata {
        extern "C" {
            fn LLVMDIBuilderCreateCompileUnit(
                builder: *const LlvmDiBuilder,
                lang: DwarfSourceLanguage,
                file: DebugFile,
                producer: *const u8,
                produer_len: usize,
                is_optimized: bool,
                flags: *const u8,
                flags_len: usize,
                runtime_version: u32,
                split_name: *const u8,
                split_name_len: usize,
                kind: DwarfEmissionKind,
                dwo_id: u32,
                split_debug_inlining: bool,
                debug_info_for_profiling: bool,
                sysroot: *const u8,
                sysroot_len: usize,
                sdk: *const u8,
                sdk_len: usize,
            ) -> Metadata;
        }

        let flags = StringRef::default();
        let split = StringRef::default();
        let sysroot = StringRef::default();
        let sdk = StringRef::default();

        unsafe {
            LLVMDIBuilderCreateCompileUnit(
                self.builder,
                DwarfSourceLanguage::C,
                file,
                PRODUCER.as_bytes().as_ptr() as *const u8,
                PRODUCER.as_bytes().len(),
                self.optimized,
                flags.data,
                flags.len,
                RUNTIME_VERSION,
                split.data,
                split.len,
                DwarfEmissionKind::Full,
                DWO_ID,
                /* splitDebugInlining= */ true,
                /* debugInfoForProfiling= */ false,
                sysroot.data,
                sysroot.len,
                sdk.data,
                sdk.len,
            )
        }
    }

    /// Create a file descriptor to hold debugging information for a file.
    pub fn create_file(&self, file: &Path) -> DebugFile {
        extern "C" {
            fn LLVMDIBuilderCreateFile(
                builder: *const LlvmDiBuilder,
                filename: *const u8,
                filename_len: usize,
                directory: *const u8,
                directory_len: usize,
            ) -> DebugFile;
        }

        let cwd = self.cwd.as_path();
        let (file, dir) = if file.is_absolute() {
            // Strip the common prefix (if it is more than just '/')
            // from current directory and filename to keep things less verbose
            if let Ok(stripped) = file.strip_prefix(cwd) {
                (stripped.file_name().unwrap(), stripped.parent())
            } else {
                (file.file_name().unwrap(), file.parent())
            }
        } else {
            (file.file_name().unwrap(), file.parent())
        };

        let filename = StringRef::from(file);
        let directory = dir.map(StringRef::from).unwrap_or_default();

        unsafe {
            LLVMDIBuilderCreateFile(
                self.builder,
                filename.data,
                filename.len,
                directory.data,
                directory.len,
            )
        }
    }

    /// Create a new descriptor for the specified subprogram.
    pub fn create_function(
        &self,
        scope: DebugScope,
        name: &str,
        linkage_name: &str,
        file: DebugFile,
        line: usize,
        ty: Metadata,
        is_local: bool,
        is_definition: bool,
        scope_line: usize,
        flags: DIFlags,
    ) -> Metadata {
        extern "C" {
            fn LLVMDIBuilderCreateFunction(
                builder: *const LlvmDiBuilder,
                scope: DebugScope,
                name: *const u8,
                name_len: usize,
                linkage_name: *const u8,
                linkage_name_len: usize,
                file: DebugFile,
                line: u32,
                ty: Metadata,
                is_local_to_unit: bool,
                is_definition: bool,
                scope_line: u32,
                flags: DIFlags,
                is_optimized: bool,
            ) -> Metadata;
        }

        let name = StringRef::from(name);
        let linkage_name = StringRef::from(linkage_name);

        unsafe {
            LLVMDIBuilderCreateFunction(
                self.builder,
                scope,
                name.data,
                name.len,
                linkage_name.data,
                linkage_name.len,
                file,
                line.try_into().unwrap(),
                ty,
                is_local,
                is_definition,
                scope_line.try_into().unwrap(),
                flags,
                self.optimized,
            )
        }
    }

    /// Create a descriptor for a lexical block with the specified parent context.
    pub fn create_lexical_block(
        &self,
        scope: DebugScope,
        file: DebugFile,
        line: usize,
        column: usize,
    ) -> Metadata {
        extern "C" {
            fn LLVMDIBuilderCreateLexicalBlock(
                builder: *const LlvmDiBuilder,
                scope: DebugScope,
                file: DebugFile,
                line: u32,
                column: u32,
            ) -> Metadata;
        }

        unsafe {
            LLVMDIBuilderCreateLexicalBlock(
                self.builder,
                scope,
                file,
                line.try_into().unwrap(),
                column.try_into().unwrap(),
            )
        }
    }

    /// Create a descriptor for an imported function, type, or variable.
    pub fn create_imported_declaration<S: Into<StringRef>>(
        &self,
        scope: DebugScope,
        decl: Metadata,
        name: S,
        file: DebugFile,
        line: usize,
    ) -> Metadata {
        extern "C" {
            fn LLVMDIBuilderCreateImportedDeclaration(
                builder: *const LlvmDiBuilder,
                scope: DebugScope,
                decl: Metadata,
                file: DebugFile,
                line: u32,
                name: *const u8,
                name_len: usize,
                elements: *const Metadata,
                num_elements: u32,
            ) -> Metadata;
        }

        let name = name.into();
        unsafe {
            LLVMDIBuilderCreateImportedDeclaration(
                self.builder,
                scope,
                decl,
                file,
                line.try_into().unwrap(),
                name.data,
                name.len,
                std::ptr::null(),
                0,
            )
        }
    }

    /// Creates a new DebugLocation that describes a source location
    ///
    /// If the item cannot be attributed to a given source line, use zeroes for the line/column
    pub fn create_location(
        &self,
        context: Context,
        line: u32,
        column: u32,
        scope: DebugScope,
        inlined_at: Option<Metadata>,
    ) -> DebugLocation {
        extern "C" {
            fn LLVMDIBuilderCreateDebugLocation(
                context: Context,
                line: u32,
                column: u32,
                scope: DebugScope,
                inlined_at: Metadata,
            ) -> DebugLocation;
        }

        unsafe {
            LLVMDIBuilderCreateDebugLocation(
                context,
                line,
                column,
                scope,
                inlined_at.unwrap_or_else(Metadata::null),
            )
        }
    }

    /// Construct any deferred debug info descriptors
    pub fn build(mut self) {
        self.finalize()
    }

    fn finalize(&mut self) {
        extern "C" {
            fn LLVMDIBuilderFinalize(builder: *const LlvmDiBuilder);
        }

        unsafe {
            LLVMDIBuilderFinalize(self.builder);
        }

        self.finalized = true;
    }
}

impl<'m> Drop for DebugInfoBuilder<'m> {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeDIBuilder(builder: *const LlvmDiBuilder);
        }

        if !self.finalized {
            self.finalize()
        }

        unsafe { LLVMDisposeDIBuilder(self.builder) }
    }
}

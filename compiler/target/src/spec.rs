pub mod abi;
mod android_base;
mod apple_base;
mod apple_sdk_base;
pub mod crt_objects;
mod dragonfly_base;
mod freebsd_base;
mod linux_base;
mod linux_gnu_base;
mod linux_musl_base;
mod msvc_base;
mod netbsd_base;
mod openbsd_base;
mod wasm_base;
mod windows_gnu_base;
mod windows_gnullvm_base;
mod windows_msvc_base;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodingType {
    /// Use the default encoding based on target pointer width
    Default,
    /// Use a 32-bit encoding
    Encoding32,
    /// Use a 64-bit encoding
    Encoding64,
    /// An alternative 64-bit encoding, based on NaN-boxing
    Encoding64Nanboxed,
}
impl EncodingType {
    #[inline(always)]
    pub fn is_nanboxed(self) -> bool {
        self == Self::Encoding64Nanboxed
    }
}

use self::abi::Abi;
use self::crt_objects::{CrtObjects, LinkSelfContainedDefault};

#[derive(Error, Debug)]
#[error("invalid linker flavor: '{0}'")]
pub struct InvalidLinkerFlavorError(String);

#[derive(Error, Debug)]
pub enum TargetError {
    #[error("unsupported target: '{0}'")]
    Unsupported(String),
    #[error("invalid target: {0}")]
    Other(String),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LinkerFlavor {
    Gcc,
    Ld,
    Lld(LldFlavor),
    Msvc,
    EmCc,
    Bpf,
    Ptx,
}
impl fmt::Display for LinkerFlavor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Gcc => f.write_str("cc"),
            Self::Ld => f.write_str("ld"),
            Self::Lld(_) => f.write_str("lld"),
            Self::Msvc => f.write_str("link.exe"),
            Self::EmCc => f.write_str("emcc"),
            Self::Bpf => f.write_str("bpf-linker"),
            Self::Ptx => f.write_str("ptx-linker"),
        }
    }
}

pub type LinkArgs = BTreeMap<LinkerFlavor, Vec<Cow<'static, str>>>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LldFlavor {
    Wasm,
    Ld64,
    Ld,
    Link,
}
impl fmt::Display for LldFlavor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Wasm => f.write_str("wasm-ld"),
            Self::Ld64 => f.write_str("ld64.lld"),
            Self::Ld => f.write_str("ld.lld"),
            Self::Link => f.write_str("link"),
        }
    }
}
impl FromStr for LldFlavor {
    type Err = InvalidLinkerFlavorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "darwin" | "ld64.lld" => Ok(LldFlavor::Ld64),
            "gnu" | "ld" | "ld.lld" => Ok(LldFlavor::Ld),
            "link" | "link.exe" => Ok(LldFlavor::Link),
            "wasm" | "wasm-ld" => Ok(LldFlavor::Wasm),
            _ => Err(InvalidLinkerFlavorError(s.to_string())),
        }
    }
}

macro_rules! flavor_mappings {
    ($((($($flavor:tt)*), $string:expr),)*) => (
        impl LinkerFlavor {
            pub const fn one_of() -> &'static str {
                concat!("one of: ", $($string, " ",)*)
            }

            pub fn desc(&self) -> &str {
                match *self {
                    $($($flavor)* => $string,)*
                }
            }
        }
        impl FromStr for LinkerFlavor {
            type Err = InvalidLinkerFlavorError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($string => Ok($($flavor)*),)*
                    _ => Err(InvalidLinkerFlavorError(s.to_string()))
                }
            }
        }
    )
}

flavor_mappings! {
    ((LinkerFlavor::Gcc), "gcc"),
    ((LinkerFlavor::Ld), "ld"),
    ((LinkerFlavor::Lld(LldFlavor::Ld)), "ld.lld"),
    ((LinkerFlavor::Lld(LldFlavor::Ld64)), "ld64.lld"),
    ((LinkerFlavor::Lld(LldFlavor::Link)), "lld-link"),
    ((LinkerFlavor::Lld(LldFlavor::Wasm)), "wasm-ld"),
    ((LinkerFlavor::Msvc), "msvc"),
    ((LinkerFlavor::EmCc), "em"),
    ((LinkerFlavor::Bpf), "bpf-linker"),
    ((LinkerFlavor::Ptx), "ptx-linker"),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum RelocModel {
    Static,
    Pic,
    DynamicNoPic,
    /// Read-Only Position Independence
    ///
    /// Code and read-only data is accessed PC-relative
    /// The offsets between all code and RO data sections are known at static link time
    Ropi,
    /// Read-Write Position Indepdendence
    ///
    /// Read-write data is accessed relative to the static base register.
    /// The offsets between all writeable data sections are known at static
    /// link time. This does not affect read-only data.
    Rwpi,
    /// A combination of both of the above
    RopiRwpi,
}
impl FromStr for RelocModel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "static" => Ok(Self::Static),
            "pic" => Ok(Self::Pic),
            "dynamic-no-pic" => Ok(Self::DynamicNoPic),
            "ropi" => Ok(Self::Ropi),
            "rwpi" => Ok(Self::Rwpi),
            "ropi-rwpi" => Ok(Self::RopiRwpi),
            _ => Err(()),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum CodeModel {
    Tiny,
    Small,
    Kernel,
    Medium,
    Large,
}
impl FromStr for CodeModel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "tiny" => Ok(Self::Tiny),
            "small" => Ok(Self::Small),
            "kernel" => Ok(Self::Kernel),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            _ => Err(()),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Hash)]
pub enum TlsModel {
    NotThreadLocal,
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec,
}
impl FromStr for TlsModel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "global-dynamic" => Ok(Self::GeneralDynamic),
            "local-dynamic" => Ok(Self::LocalDynamic),
            "initial-exec" => Ok(Self::InitialExec),
            "local-exec" => Ok(Self::LocalExec),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum PanicStrategy {
    Unwind,
    Abort,
}
impl fmt::Display for PanicStrategy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Unwind => f.write_str("unwind"),
            Self::Abort => f.write_str("abort"),
        }
    }
}
impl FromStr for PanicStrategy {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "unwind" => Ok(Self::Unwind),
            "abort" => Ok(Self::Abort),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum RelroLevel {
    Full,
    Partial,
    Off,
    None,
}

impl RelroLevel {
    pub fn desc(&self) -> &str {
        match *self {
            RelroLevel::Full => "full",
            RelroLevel::Partial => "partial",
            RelroLevel::Off => "off",
            RelroLevel::None => "none",
        }
    }
}

impl FromStr for RelroLevel {
    type Err = ();

    fn from_str(s: &str) -> Result<RelroLevel, ()> {
        match s {
            "full" => Ok(RelroLevel::Full),
            "partial" => Ok(RelroLevel::Partial),
            "off" => Ok(RelroLevel::Off),
            "none" => Ok(RelroLevel::None),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LinkOutputKind {
    /// Dynamically linked non position-independent executable.
    DynamicNoPicExe,
    /// Dynamically linked position-independent executable.
    DynamicPicExe,
    /// Statically linked non position-independent executable.
    StaticNoPicExe,
    /// Statically linked position-independent executable.
    StaticPicExe,
    /// Regular dynamic library ("dynamically linked").
    DynamicDylib,
    /// Dynamic library with bundled libc ("statically linked").
    StaticDylib,
    /// WASI module with a lifetime past the _initialize entry point
    WasiReactorExe,
}

impl LinkOutputKind {
    fn as_str(&self) -> &'static str {
        match self {
            LinkOutputKind::DynamicNoPicExe => "dynamic-nopic-exe",
            LinkOutputKind::DynamicPicExe => "dynamic-pic-exe",
            LinkOutputKind::StaticNoPicExe => "static-nopic-exe",
            LinkOutputKind::StaticPicExe => "static-pic-exe",
            LinkOutputKind::DynamicDylib => "dynamic-dylib",
            LinkOutputKind::StaticDylib => "static-dylib",
            LinkOutputKind::WasiReactorExe => "wasi-reactor-exe",
        }
    }

    pub fn from_str(s: &str) -> Option<LinkOutputKind> {
        Some(match s {
            "dynamic-nopic-exe" => LinkOutputKind::DynamicNoPicExe,
            "dynamic-pic-exe" => LinkOutputKind::DynamicPicExe,
            "static-nopic-exe" => LinkOutputKind::StaticNoPicExe,
            "static-pic-exe" => LinkOutputKind::StaticPicExe,
            "dynamic-dylib" => LinkOutputKind::DynamicDylib,
            "static-dylib" => LinkOutputKind::StaticDylib,
            "wasi-reactor-exe" => LinkOutputKind::WasiReactorExe,
            _ => return None,
        })
    }
}

impl fmt::Display for LinkOutputKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DebuginfoKind {
    /// DWARF debuginfo (such as that used on `x86_64_unknown_linux_gnu`).
    #[default]
    Dwarf,
    /// DWARF debuginfo in dSYM files (such as on Apple platforms).
    DwarfDsym,
    /// Program database files (such as on Windows).
    Pdb,
}
impl DebuginfoKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dwarf => "dwarf",
            Self::DwarfDsym => "dwarf-dsym",
            Self::Pdb => "pdb",
        }
    }
}
impl fmt::Display for DebuginfoKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
impl FromStr for DebuginfoKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dwarf" => Ok(Self::Dwarf),
            "dwarf-dsym" => Ok(Self::DwarfDsym),
            "pdb" => Ok(Self::Pdb),
            _ => Err(()),
        }
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitDebugInfo {
    /// Split debug-information is disabled, meaning that on supported platforms
    /// you can find all debug information in the executable itself. This is
    /// only supported for ELF effectively.
    ///
    /// * Windows - not supported
    /// * macOS - don't run `dsymutil`
    /// * ELF - `.dwarf_*` sections
    #[default]
    Off,
    /// Split debug-information can be found in a "packed" location separate
    /// from the final artifact. This is supported on all platforms.
    ///
    /// * Windows - `*.pdb`
    /// * macOS - `*.dSYM` (run `dsymutil`)
    /// * ELF - `*.dwp` (run `rust-llvm-dwp`)
    Packed,
    /// Split debug-information can be found in individual object files on the
    /// filesystem. The main executable may point to the object files.
    ///
    /// * Windows - not supported
    /// * macOS - supported, scattered object files
    /// * ELF - supported, scattered `*.dwo` or `*.o` files (see `SplitDwarfKind`)
    Unpacked,
}
impl fmt::Display for SplitDebugInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Off => f.write_str("off"),
            Self::Packed => f.write_str("packed"),
            Self::Unpacked => f.write_str("unpacked"),
        }
    }
}
impl FromStr for SplitDebugInfo {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "off" => Ok(Self::Off),
            "packed" => Ok(Self::Packed),
            "unpacked" => Ok(Self::Unpacked),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum FramePointer {
    /// Forces the machine code generator to always preserve the frame pointers.
    Always,
    /// Forces the machine code generator to preserve the frame pointers except for the leaf
    /// functions (i.e. those that don't call other functions).
    NonLeaf,
    /// Allows the machine code generator to omit the frame pointers.
    ///
    /// This option does not guarantee that the frame pointers will be omitted.
    MayOmit,
}
impl fmt::Display for FramePointer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Always => f.write_str("always"),
            Self::NonLeaf => f.write_str("non-leaf"),
            Self::MayOmit => f.write_str("may-omit"),
        }
    }
}
impl FromStr for FramePointer {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        Ok(match s {
            "always" => Self::Always,
            "non-leaf" => Self::NonLeaf,
            "may-omit" => Self::MayOmit,
            _ => return Err(()),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum MergeFunctions {
    Disabled,
    Trampolines,
    Aliases,
}

impl MergeFunctions {
    pub fn desc(&self) -> &str {
        match *self {
            MergeFunctions::Disabled => "disabled",
            MergeFunctions::Trampolines => "trampolines",
            MergeFunctions::Aliases => "aliases",
        }
    }
}

impl FromStr for MergeFunctions {
    type Err = ();

    fn from_str(s: &str) -> Result<MergeFunctions, ()> {
        match s {
            "disabled" => Ok(MergeFunctions::Disabled),
            "trampolines" => Ok(MergeFunctions::Trampolines),
            "aliases" => Ok(MergeFunctions::Aliases),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StackProbeType {
    /// Don't emit any stack probes.
    None,
    /// It is harmless to use this option even on targets that do not have backend support for
    /// stack probes as the failure mode is the same as if no stack-probe option was specified in
    /// the first place.
    Inline,
    /// Call `__rust_probestack` whenever stack needs to be probed.
    Call,
    /// Use inline option for LLVM versions later than specified in `min_llvm_version_for_inline`
    /// and call `__rust_probestack` otherwise.
    InlineOrCall {
        min_llvm_version_for_inline: (u32, u32, u32),
    },
}

macro_rules! supported_targets {
    ( $(($( $triple:literal, )+ $module:ident ),)+ ) => {
        $(mod $module;)+

        /// List of supported targets
        const TARGETS: &[&str] = &[$($($triple),+),+];

        fn search<S: AsRef<str>>(target: S) -> Result<Target, TargetError> {
            let target = target.as_ref();
            match target {
                $(
                    $($triple)|+ => Ok($module::target()),
                )+
                    _ => Err(TargetError::Unsupported(target.to_string())),
            }
        }

        pub fn get_targets() -> impl Iterator<Item = String> {
            TARGETS.iter().filter_map(|t| -> Option<String> {
                search(t)
                    .and(Ok(t.to_string()))
                    .ok()
            })
        }
    };
}

supported_targets! {
    ("armv7-apple-ios", armv7_apple_ios),
    ("armv7s-apple-ios", armv7s_apple_ios),
    ("armv7k-apple-watchos", armv7k_apple_watchos),

    ("aarch64-apple-darwin", aarch64_apple_darwin),
    ("aarch64-apple-ios", aarch64_apple_ios),
    ("aarch64-apple-ios-macabi", aarch64_apple_ios_macabi),
    ("aarch64-apple-ios-sim", aarch64_apple_ios_sim),
    ("aarch64-apple-tvos", aarch64_apple_tvos),
    ("aarch64-apple-watchos-sim", aarch64_apple_watchos_sim),

    ("aarch64-linux-android", aarch64_linux_android),

    ("aarch64-unknown-linux-gnu", aarch64_unknown_linux_gnu),
    ("aarch64-unknown-linux-musl", aarch64_unknown_linux_musl),

    ("aarch64-unknown-freebsd", aarch64_unknown_freebsd),
    ("aarch64-unknown-netbsd", aarch64_unknown_netbsd),
    ("aarch64-unknown-openbsd", aarch64_unknown_openbsd),

    ("i686-apple-darwin", i686_apple_darwin),

    ("i686-linux-android", i686_linux_android),

    ("i686-unknown-linux-gnu", i686_unknown_linux_gnu),
    ("i686-unknown-linux-musl", i686_unknown_linux_musl),

    ("i686-unknown-freebsd", i686_unknown_freebsd),
    ("i686-unknown-netbsd", i686_unknown_netbsd),
    ("i686-unknown-openbsd", i686_unknown_openbsd),

    ("wasm32-unknown-emscripten", wasm32_unknown_emscripten),
    ("wasm32-unknown-unknown", wasm32_unknown_unknown),
    ("wasm32-wasi", wasm32_wasi),
    ("wasm64-unknown-unknown", wasm64_unknown_unknown),

    ("x86_64-apple-darwin", x86_64_apple_darwin),
    ("x86_64-apple-ios", x86_64_apple_ios),
    ("x86_64-apple-ios-macabi", x86_64_apple_ios_macabi),
    ("x86_64-apple-tvos", x86_64_apple_tvos),
    ("x86_64-apple-watchos-sim", x86_64_apple_watchos_sim),

    ("x86_64-linux-android", x86_64_linux_android),

    ("x86_64-unknown-linux-gnu", x86_64_unknown_linux_gnu),
    ("x86_64-unknown-linux-musl", x86_64_unknown_linux_musl),

    ("x86_64-unknown-dragonfly", x86_64_unknown_dragonfly),
    ("x86_64-unknown-freebsd", x86_64_unknown_freebsd),
    ("x86_64-unknown-netbsd", x86_64_unknown_netbsd),
    ("x86_64-unknown-openbsd", x86_64_unknown_openbsd),

    ("x86_64-pc-windows-msvc", x86_64_pc_windows_msvc),
    ("x86_64-pc-windows-gnu", x86_64_pc_windows_gnu),
    ("x86_64-pc-windows-gnullvm", x86_64_pc_windows_gnullvm),
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Endianness {
    Big,
    Little,
    Native,
}
impl Default for Endianness {
    fn default() -> Self {
        Self::Native
    }
}
impl ToString for Endianness {
    fn to_string(&self) -> String {
        match *self {
            Self::Big => "big".to_string(),
            Self::Little => "little".to_string(),
            Self::Native => "native".to_string(),
        }
    }
}

/// Everything Firefly knows about how to compile for a specific target.
///
/// Every field here must be specified, and has no default value.
#[derive(PartialEq, Clone, Debug)]
pub struct Target {
    /// Target triple to pass to LLVM.
    pub llvm_target: Cow<'static, str>,
    /// The bit width of target pointers
    pub pointer_width: usize,
    /// Architecture to use for ABI considerations. Valid options include: "x86",
    /// "x86_64", "arm", "aarch64", "mips", "powerpc", "powerpc64", and others.
    pub arch: Cow<'static, str>,
    /// [Data layout](http://llvm.org/docs/LangRef.html#data-layout) to pass to LLVM.
    pub data_layout: Cow<'static, str>,
    /// Optional settings with defaults.
    pub options: TargetOptions,
}
impl Target {
    pub fn triple(&self) -> &str {
        use std::borrow::Borrow;
        self.llvm_target.borrow()
    }

    pub fn search(target: &str) -> Result<Target, TargetError> {
        self::search(target)
    }

    pub fn all() -> impl Iterator<Item = String> {
        self::get_targets()
    }

    /// Returns the term encoding used on this target
    pub fn term_encoding(&self) -> EncodingType {
        self.options.encoding
    }
}

pub trait HasTargetSpec {
    fn target_spec(&self) -> &Target;
}

impl HasTargetSpec for Target {
    fn target_spec(&self) -> &Target {
        self
    }
}

/// Optional aspects of a target specification.
///
/// This has an implementation of `Default`, see each field for what the default is. In general,
/// these try to take "minimal defaults" that don't assume anything about the runtime they run in.
#[derive(PartialEq, Clone, Debug)]
pub struct TargetOptions {
    /// Is this target built-in or loaded from a custom spec
    pub is_builtin: bool,
    /// Term encoding
    pub encoding: EncodingType,
    /// The endianness of the target
    pub endianness: Endianness,
    /// The width of c_int type. Defaults to "32".
    pub c_int_width: Cow<'static, str>,
    /// OS name to use for conditional compilation. Defaults to "none".
    /// "none" implies a bare metal target without a standard library.
    pub os: Cow<'static, str>,
    /// Environment name to use for conditional compilation. Defaults to "".
    pub env: Cow<'static, str>,
    /// ABI name to distinguish multiple ABIs on the same OS and architecture, e.g. "eabihf". Defaults to "".
    pub abi: Cow<'static, str>,
    /// Vendor name to use for conditional compilation. Defaults to "unknown".
    pub vendor: Cow<'static, str>,
    /// Default linker flavor used if `-C linker-flavor` or `-C linker` are not passed on the command line.
    /// Defaults to `LinkerFlavor::Gcc`.
    pub linker_flavor: LinkerFlavor,
    /// Linker to invoke
    pub linker: Option<Cow<'static, str>>,
    /// LLD flavor used if `lld` (or `firefly-lld`) is specified as a linker without clarifying its flavor
    pub lld_flavor: LldFlavor,
    /// Whether the linker support GNU-like arguments such as -O. Defaults to false.
    pub linker_is_gnu: bool,

    /// Objects to link before and after all other object code.
    pub pre_link_objects: CrtObjects,
    pub post_link_objects: CrtObjects,
    /// Same as `(pre|post)_link_objects`, but when we fail to pull the objects with help of the
    /// target's native gcc and fall back to the "self-contained" mode and pull them manually.
    /// See `crt_objects.rs` for some more detailed documentation.
    pub pre_link_objects_self_contained: CrtObjects,
    pub post_link_objects_self_contained: CrtObjects,
    /// Which logic to use to determine whether to fall back to the "self-contained" mode or not.
    pub link_self_contained: LinkSelfContainedDefault,

    /// Linker arguments that are passed *before* any user-defined libraries.
    pub pre_link_args: LinkArgs, // ... unconditionally
    /// Linker arguments that are unconditionally passed after any
    /// user-defined but before post_link_objects. Standard platform
    /// libraries that should be always be linked to, usually go here.
    pub late_link_args: LinkArgs,
    /// Linker arguments used in addition to `late_link_args` if at least one
    /// dependency is dynamically linked.
    pub late_link_args_dynamic: LinkArgs,
    /// Linker arguments used in addition to `late_link_args` if all
    /// dependencies are statically linked.
    pub late_link_args_static: LinkArgs,
    /// Linker arguments that are unconditionally passed *after* any
    /// user-defined libraries.
    pub post_link_args: LinkArgs,

    /// Optional link script applied to `dylib` and `executable` crate types.
    /// This is a string containing the script, not a path. Can only be applied
    /// to linkers where `linker_is_gnu` is true.
    pub link_script: Option<Cow<'static, str>>,
    /// Environment variables to be set for the linker invocation.
    pub link_env: Vec<(Cow<'static, str>, Cow<'static, str>)>,
    /// Environment variables to be removed for the linker invocation.
    pub link_env_remove: Vec<Cow<'static, str>>,

    /// Extra arguments to pass to the external assembler (when used)
    pub asm_args: Vec<Cow<'static, str>>,

    /// Default CPU to pass to LLVM. Corresponds to `llc -mcpu=$cpu`. Defaults
    /// to "generic".
    pub cpu: Cow<'static, str>,
    /// Default target features to pass to LLVM. These features will *always* be
    /// passed, and cannot be disabled even via `-C`. Corresponds to `llc
    /// -mattr=$features`.
    pub features: Cow<'static, str>,
    /// Whether dynamic linking is available on this target. Defaults to false.
    pub dynamic_linking: bool,
    /// If dynamic linking is available, whether only cdylibs are supported.
    pub only_cdylib: bool,
    /// Whether executables are available on this target. iOS, for example, only allows static
    /// libraries. Defaults to true.
    pub executables: bool,
    /// Relocation model to use in object file. Corresponds to `llc
    /// -relocation-model=$relocation_model`. Defaults to "pic".
    pub relocation_model: RelocModel,
    /// Code model to use. Corresponds to `llc -code-model=$code_model`.
    pub code_model: Option<CodeModel>,
    /// TLS model to use. Options are "global-dynamic" (default), "local-dynamic", "initial-exec"
    /// and "local-exec". This is similar to the -ftls-model option in GCC/Clang.
    pub tls_model: TlsModel,
    /// Do not emit code that uses the "red zone", if the ABI has one. Defaults to false.
    pub disable_redzone: bool,
    /// Frame pointer mode for this target. Defaults to `MayOmit`
    pub frame_pointer: FramePointer,
    /// Emit each function in its own section. Defaults to true.
    pub function_sections: bool,
    /// String to prepend to the name of every dynamic library. Defaults to "lib".
    pub dll_prefix: Cow<'static, str>,
    /// String to append to the name of every dynamic library. Defaults to ".so".
    pub dll_suffix: Cow<'static, str>,
    /// String to append to the name of every executable.
    pub exe_suffix: Cow<'static, str>,
    /// String to prepend to the name of every static library. Defaults to "lib".
    pub staticlib_prefix: Cow<'static, str>,
    /// String to append to the name of every static library. Defaults to ".a".
    pub staticlib_suffix: Cow<'static, str>,
    /// OS family to use for conditional compilation. Valid options: "unix", "windows".
    pub families: Vec<Cow<'static, str>>,
    /// Whether the target toolchain's ABI supports returning small structs as an integer.
    pub abi_return_struct_as_int: bool,
    /// Whether the target toolchain is like macOS's. Only useful for compiling against iOS/macOS,
    /// in particular running dsymutil and some other stuff like `-dead_strip`. Defaults to false.
    pub is_like_osx: bool,
    /// Whether the target toolchain is like Solaris's.
    /// Only useful for compiling against Illumos/Solaris,
    /// as they have a different set of linker flags. Defaults to false.
    pub is_like_solaris: bool,
    /// Whether the target is like Windows.
    /// This is a combination of several more specific properties represented as a single flag:
    ///   - The target uses a Windows ABI,
    ///   - uses PE/COFF as a format for object code,
    ///   - uses Windows-style dllexport/dllimport for shared libraries,
    ///   - uses import libraries and .def files for symbol exports,
    ///   - executables support setting a subsystem.
    pub is_like_windows: bool,
    /// Whether the target is like MSVC.
    /// This is a combination of several more specific properties represented as a single flag:
    ///   - The target has all the properties from `is_like_windows`
    ///     (for in-tree targets "is_like_msvc ⇒ is_like_windows" is ensured by a unit test),
    ///   - has some MSVC-specific Windows ABI properties,
    ///   - uses a link.exe-like linker,
    ///   - uses CodeView/PDB for debuginfo and natvis for its visualization,
    ///   - uses SEH-based unwinding,
    ///   - supports control flow guard mechanism.
    pub is_like_msvc: bool,
    /// Whether a target toolchain is like WASM.
    pub is_like_wasm: bool,
    /// Default supported Version of DWARF on this platform.
    /// Useful because some platforms (osx, bsd) only want up to DWARF2
    pub default_dwarf_version: u32,
    /// The MinGW toolchain has a known issue that prevents it from correctly
    /// handling COFF object files with more than 2<sup>15</sup> sections. Since each weak
    /// symbol needs its own COMDAT section, weak linkage implies a large
    /// number sections that easily exceeds the given limit for larger
    /// codebases. Consequently we want a way to disallow weak linkage on some
    /// platforms.
    pub allows_weak_linkage: bool,
    /// Whether the linker support rpaths or not. Defaults to false.
    pub has_rpath: bool,
    /// Whether to disable linking to the default libraries, typically corresponds
    /// to `-nodefaultlibs`. Defaults to true.
    pub no_default_libraries: bool,
    /// Dynamically linked executables can be compiled as position independent
    /// if the default relocation model of position independent code is not
    /// changed. This is a requirement to take advantage of ASLR, as otherwise
    /// the functions in the executable are not randomized and can be used
    /// during an exploit of a vulnerability in any code.
    pub position_independent_executables: bool,
    /// Executables that are both statically linked and position-independent are supported.
    pub static_position_independent_executables: bool,
    /// Determines if the target always requires using the PLT for indirect
    /// library calls or not. This controls the default value of the `-Z plt` flag.
    pub needs_plt: bool,
    /// Either partial, full, or off. Full RELRO makes the dynamic linker
    /// resolve all symbols at startup and marks the GOT read-only before
    /// starting the program, preventing overwriting the GOT.
    pub relro_level: RelroLevel,
    /// Format that archives should be emitted in. This affects whether we use
    /// LLVM to assemble an archive or fall back to the system linker, and
    /// currently only "gnu" is used to fall into LLVM. Unknown strings cause
    /// the system linker to be used.
    pub archive_format: Cow<'static, str>,
    /// Is asm!() allowed? Defaults to true.
    pub allow_asm: bool,
    /// Whether the runtime startup code requires the `main` function be passed
    /// `argc` and `argv` values.
    pub main_needs_argc_argv: bool,

    /// Flag indicating whether ELF TLS (e.g., #[thread_local]) is available for
    /// this target.
    pub has_thread_local: bool,
    // This is mainly for easy compatibility with emscripten.
    // If we give emcc .o files that are actually .bc files it
    // will 'just work'.
    pub obj_is_bitcode: bool,
    /// Whether the target requires that emitted object code includes bitcode.
    pub forces_embed_bitcode: bool,
    /// Content of the LLVM cmdline section associated with embedded bitcode.
    pub bitcode_llvm_cmdline: Cow<'static, str>,

    /// Don't use this field; instead use the `.min_atomic_width()` method.
    pub min_atomic_width: Option<u64>,

    /// Don't use this field; instead use the `.max_atomic_width()` method.
    pub max_atomic_width: Option<u64>,

    /// Whether the target supports atomic CAS operations natively
    pub atomic_cas: bool,

    /// Panic strategy: "unwind" or "abort"
    pub panic_strategy: PanicStrategy,

    /// Whether or not linking dylibs to a static CRT is allowed.
    pub crt_static_allows_dylibs: bool,
    /// Whether or not the CRT is statically linked by default.
    pub crt_static_default: bool,
    /// Whether or not crt-static is respected by the compiler (or is a no-op).
    pub crt_static_respected: bool,

    /// Whether or not stack probes (__rust_probestack) are enabled
    pub stack_probes: StackProbeType,

    /// The minimum alignment for global symbols.
    pub min_global_align: Option<u64>,

    /// Default number of codegen units to use in debug mode
    pub default_codegen_units: Option<u64>,

    /// Whether to generate trap instructions in places where optimization would
    /// otherwise produce control flow that falls through into unrelated memory.
    pub trap_unreachable: bool,

    /// This target requires everything to be compiled with LTO to emit a final
    /// executable, aka there is no native linker for this target.
    pub requires_lto: bool,

    /// This target has no support for threads.
    pub singlethread: bool,

    /// Whether library functions call lowering/optimization is disabled in LLVM
    /// for this target unconditionally.
    pub no_builtins: bool,

    /// The default visibility for symbols in this target should be "hidden"
    /// rather than "default"
    pub default_hidden_visibility: bool,

    /// Whether a .debug_gdb_scripts section will be added to the output object file
    pub emit_debug_gdb_scripts: bool,

    /// Whether or not to unconditionally `uwtable` attributes on functions,
    /// typically because the platform needs to unwind for things like stack
    /// unwinders.
    pub requires_uwtable: bool,

    /// Whether or not to emit `uwtable` attributes on functions if `-C force-unwind-tables`
    /// is not specified and `uwtable` is not required on this target.
    pub default_uwtable: bool,

    /// Whether or not SIMD types are passed by reference in the Rust ABI,
    /// typically required if a target can be compiled with a mixed set of
    /// target features. This is `true` by default, and `false` for targets like
    /// wasm32 where the whole program either has simd or not.
    pub simd_types_indirect: bool,

    /// Pass a list of symbol which should be exported in the dylib to the linker.
    pub limit_rdylib_exports: bool,

    /// If set, have the linker export exactly these symbols, instead of using
    /// the usual logic to figure this out from the crate itself.
    pub override_export_symbols: Option<Vec<Cow<'static, str>>>,

    /// Determines how or whether the MergeFunctions LLVM pass should run for
    /// this target. Either "disabled", "trampolines", or "aliases".
    /// The MergeFunctions pass is generally useful, but some targets may need
    /// to opt out. The default is "aliases".
    ///
    /// Workaround for: https://github.com/rust-lang/rust/issues/57356
    pub merge_functions: MergeFunctions,

    /// Use platform dependent mcount function
    pub mcount: Cow<'static, str>,

    /// LLVM ABI name, corresponds to the '-mabi' parameter available in multilib C compilers
    pub llvm_abiname: Cow<'static, str>,

    /// Whether or not RelaxElfRelocation flag will be passed to the linker
    pub relax_elf_relocations: bool,

    /// Additional arguments to pass to LLVM, similar to the `-C llvm-args` codegen option.
    pub llvm_args: Vec<Cow<'static, str>>,

    /// Whether to use legacy .ctors initialization hooks rather than .init_array. Defaults
    /// to false (uses .init_array).
    pub use_ctors_section: bool,

    /// Whether the linker is instructed to add a `GNU_EH_FRAME` ELF header
    /// used to locate unwinding information is passed
    /// (only has effect if the linker is `ld`-like).
    pub eh_frame_header: bool,

    /// Is true if the target is an ARM architecture using thumb v1 which allows for
    /// thumb and arm interworking.
    pub has_thumb_interworking: bool,

    /// Which kind of debuginfo is used by this target?
    pub debuginfo_kind: DebuginfoKind,
    /// How to handle split debug information, if at all. Specifying `None` has
    /// target-specific meaning.
    pub split_debuginfo: SplitDebugInfo,
    /// Which kinds of split debuginfo are supported by the target?
    pub supported_split_debuginfo: Cow<'static, [SplitDebugInfo]>,

    /// If present it's a default value to use for adjusting the C ABI.
    pub default_adjusted_cabi: Option<Abi>,

    /// Minimum number of bits in #[repr(C)] enum. Defaults to 32.
    pub c_enum_min_bits: u64,

    /// Whether or not the DWARF `.debug_aranges` section should be generated.
    pub generate_arange_section: bool,

    /// Whether the target supports stack canary checks. `true` by default,
    /// since this is most common among tier 1 and tier 2 targets.
    pub supports_stack_protector: bool,
}

impl Default for TargetOptions {
    /// Creates a set of "sane defaults" for any target. This is still
    /// incomplete, and if used for compilation, will certainly not work.
    fn default() -> TargetOptions {
        TargetOptions {
            is_builtin: false,
            encoding: EncodingType::Encoding64Nanboxed,
            endianness: Endianness::Little,
            c_int_width: "32".into(),
            os: "none".into(),
            env: "".into(),
            abi: "".into(),
            vendor: "unknown".into(),
            linker_flavor: LinkerFlavor::Gcc,
            linker: option_env!("CFG_DEFAULT_LINKER").map(|s| s.into()),
            lld_flavor: LldFlavor::Ld,
            linker_is_gnu: true,
            link_script: None,
            asm_args: Vec::new(),
            cpu: "generic".into(),
            features: "".into(),
            dynamic_linking: false,
            only_cdylib: false,
            executables: true,
            relocation_model: RelocModel::Pic,
            code_model: None,
            tls_model: TlsModel::GeneralDynamic,
            disable_redzone: false,
            frame_pointer: FramePointer::MayOmit,
            function_sections: true,
            dll_prefix: "lib".into(),
            dll_suffix: ".so".into(),
            exe_suffix: "".into(),
            staticlib_prefix: "lib".into(),
            staticlib_suffix: ".a".into(),
            families: vec![],
            abi_return_struct_as_int: false,
            is_like_osx: false,
            is_like_solaris: false,
            is_like_windows: false,
            is_like_msvc: false,
            is_like_wasm: false,
            default_dwarf_version: 4,
            allows_weak_linkage: true,
            has_rpath: false,
            no_default_libraries: true,
            position_independent_executables: false,
            static_position_independent_executables: false,
            needs_plt: false,
            relro_level: RelroLevel::None,
            pre_link_objects: Default::default(),
            post_link_objects: Default::default(),
            pre_link_objects_self_contained: Default::default(),
            post_link_objects_self_contained: Default::default(),
            link_self_contained: LinkSelfContainedDefault::False,
            pre_link_args: LinkArgs::new(),
            post_link_args: LinkArgs::new(),
            late_link_args: LinkArgs::new(),
            late_link_args_dynamic: LinkArgs::new(),
            late_link_args_static: LinkArgs::new(),
            link_env: Vec::new(),
            link_env_remove: Vec::new(),
            archive_format: "gnu".into(),
            main_needs_argc_argv: true,
            allow_asm: true,
            has_thread_local: false,
            obj_is_bitcode: false,
            forces_embed_bitcode: false,
            bitcode_llvm_cmdline: "".into(),
            min_atomic_width: None,
            max_atomic_width: None,
            atomic_cas: true,
            panic_strategy: PanicStrategy::Unwind,
            crt_static_allows_dylibs: false,
            crt_static_default: false,
            crt_static_respected: false,
            stack_probes: StackProbeType::None,
            min_global_align: None,
            default_codegen_units: None,
            trap_unreachable: true,
            requires_lto: false,
            singlethread: false,
            no_builtins: false,
            default_hidden_visibility: false,
            emit_debug_gdb_scripts: true,
            requires_uwtable: false,
            default_uwtable: false,
            simd_types_indirect: true,
            limit_rdylib_exports: true,
            override_export_symbols: None,
            merge_functions: MergeFunctions::Aliases,
            mcount: "mcount".into(),
            llvm_abiname: "".into(),
            relax_elf_relocations: false,
            llvm_args: vec![],
            use_ctors_section: false,
            eh_frame_header: true,
            has_thumb_interworking: false,
            debuginfo_kind: Default::default(),
            split_debuginfo: Default::default(),
            supported_split_debuginfo: Cow::Borrowed(&[SplitDebugInfo::Off]),
            default_adjusted_cabi: None,
            c_enum_min_bits: 32,
            generate_arange_section: true,
            supports_stack_protector: true,
        }
    }
}

/// Add arguments for the given flavor and also for its "twin" flavors
/// that have a compatible command line interface.
fn add_link_args(link_args: &mut LinkArgs, flavor: LinkerFlavor, args: &[&'static str]) {
    let mut insert = |flavor| {
        link_args
            .entry(flavor)
            .or_default()
            .extend(args.iter().copied().map(Cow::Borrowed))
    };
    insert(flavor);
    match flavor {
        LinkerFlavor::Ld => insert(LinkerFlavor::Lld(LldFlavor::Ld)),
        LinkerFlavor::Msvc => insert(LinkerFlavor::Lld(LldFlavor::Link)),
        LinkerFlavor::Lld(LldFlavor::Ld64) | LinkerFlavor::Lld(LldFlavor::Wasm) => {}
        LinkerFlavor::Lld(lld_flavor) => {
            panic!("add_link_args: use non-LLD flavor for {:?}", lld_flavor)
        }
        LinkerFlavor::Gcc | LinkerFlavor::EmCc | LinkerFlavor::Bpf | LinkerFlavor::Ptx => {}
    }
}

impl TargetOptions {
    fn link_args(flavor: LinkerFlavor, args: &[&'static str]) -> LinkArgs {
        let mut link_args = LinkArgs::new();
        add_link_args(&mut link_args, flavor, args);
        link_args
    }

    fn add_pre_link_args(&mut self, flavor: LinkerFlavor, args: &[&'static str]) {
        add_link_args(&mut self.pre_link_args, flavor, args);
    }

    #[allow(unused)]
    fn add_post_link_args(&mut self, flavor: LinkerFlavor, args: &[&'static str]) {
        add_link_args(&mut self.post_link_args, flavor, args);
    }
}

impl Target {
    /// Given a function ABI, turn it into the correct ABI for this target.
    pub fn adjust_abi(&self, abi: Abi) -> Abi {
        match abi {
            Abi::C { .. } => self.options.default_adjusted_cabi.unwrap_or(abi),
            Abi::System { unwind } if self.options.is_like_windows && self.arch == "x86" => {
                Abi::Stdcall { unwind }
            }
            Abi::System { unwind } => Abi::C { unwind },
            Abi::EfiApi if self.arch == "x86_64" => Abi::Win64 { unwind: false },
            Abi::EfiApi => Abi::C { unwind: false },

            // See commentary in `is_abi_supported`.
            Abi::Stdcall { .. } | Abi::Thiscall { .. } if self.arch == "x86" => abi,
            Abi::Stdcall { unwind } | Abi::Thiscall { unwind } => Abi::C { unwind },
            Abi::Fastcall { .. } if self.arch == "x86" => abi,
            Abi::Vectorcall { .. } if ["x86", "x86_64"].contains(&&self.arch[..]) => abi,
            Abi::Fastcall { unwind } | Abi::Vectorcall { unwind } => Abi::C { unwind },

            abi => abi,
        }
    }

    /// Minimum integer size in bits that this target can perform atomic
    /// operations on.
    pub fn min_atomic_width(&self) -> u64 {
        self.options.min_atomic_width.unwrap_or(8)
    }

    /// Maximum integer size in bits that this target can perform atomic
    /// operations on.
    pub fn max_atomic_width(&self) -> u64 {
        self.options
            .max_atomic_width
            .unwrap_or_else(|| self.pointer_width as u64)
    }

    /// Returns a None if the UNSUPPORTED_CALLING_CONVENTIONS lint should be emitted
    pub fn is_abi_supported(&self, abi: Abi) -> Option<bool> {
        use Abi::*;
        Some(match abi {
            Erlang
            | Rust
            | C { .. }
            | System { .. }
            | RustIntrinsic
            | RustCall
            | PlatformIntrinsic
            | Unadjusted
            | Cdecl { .. }
            | EfiApi
            | RustCold => true,
            X86Interrupt => ["x86", "x86_64"].contains(&&self.arch[..]),
            Aapcs { .. } => "arm" == self.arch,
            CCmseNonSecureCall => ["arm", "aarch64"].contains(&&self.arch[..]),
            Win64 { .. } | SysV64 { .. } => self.arch == "x86_64",
            PtxKernel => self.arch == "nvptx64",
            Msp430Interrupt => self.arch == "msp430",
            AmdGpuKernel => self.arch == "amdgcn",
            AvrInterrupt | AvrNonBlockingInterrupt => self.arch == "avr",
            Wasm => ["wasm32", "wasm64"].contains(&&self.arch[..]),
            Thiscall { .. } => self.arch == "x86",
            // On windows these fall-back to platform native calling convention (C) when the
            // architecture is not supported.
            //
            // This is I believe a historical accident that has occurred as part of Microsoft
            // striving to allow most of the code to "just" compile when support for 64-bit x86
            // was added and then later again, when support for ARM architectures was added.
            //
            // This is well documented across MSDN. Support for this in Rust has been added in
            // #54576. This makes much more sense in context of Microsoft's C++ than it does in
            // Rust, but there isn't much leeway remaining here to change it back at the time this
            // comment has been written.
            //
            // Following are the relevant excerpts from the MSDN documentation.
            //
            // > The __vectorcall calling convention is only supported in native code on x86 and
            // x64 processors that include Streaming SIMD Extensions 2 (SSE2) and above.
            // > ...
            // > On ARM machines, __vectorcall is accepted and ignored by the compiler.
            //
            // -- https://docs.microsoft.com/en-us/cpp/cpp/vectorcall?view=msvc-160
            //
            // > On ARM and x64 processors, __stdcall is accepted and ignored by the compiler;
            //
            // -- https://docs.microsoft.com/en-us/cpp/cpp/stdcall?view=msvc-160
            //
            // > In most cases, keywords or compiler switches that specify an unsupported
            // > convention on a particular platform are ignored, and the platform default
            // > convention is used.
            //
            // -- https://docs.microsoft.com/en-us/cpp/cpp/argument-passing-and-naming-conventions
            Stdcall { .. } | Fastcall { .. } | Vectorcall { .. }
                if self.options.is_like_windows =>
            {
                true
            }
            // Outside of Windows we want to only support these calling conventions for the
            // architectures for which these calling conventions are actually well defined.
            Stdcall { .. } | Fastcall { .. } if self.arch == "x86" => true,
            Vectorcall { .. } if ["x86", "x86_64"].contains(&&self.arch[..]) => true,
            // Return a `None` for other cases so that we know to emit a future compat lint.
            Stdcall { .. } | Fastcall { .. } | Vectorcall { .. } => return None,
        })
    }
}

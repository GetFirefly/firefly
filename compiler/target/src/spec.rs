pub mod abi;
mod android_base;
mod apple_base;
mod apple_sdk_base;
mod arm_base;
mod cloudabi_base;
mod dragonfly_base;
mod freebsd_base;
mod fuchsia_base;
mod haiku_base;
mod hermit_base;
mod hermit_kernel_base;
mod l4re_base;
mod linux_base;
mod linux_kernel_base;
mod linux_musl_base;
mod netbsd_base;
mod openbsd_base;
mod redox_base;
mod riscv_base;
mod solaris_base;
mod thumb_base;
mod uefi_base;
mod vxworks_base;
mod wasm32_base;
mod windows_base;
mod windows_msvc_base;
mod windows_uwp_base;
mod windows_uwp_msvc_base;

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use thiserror::Error;

pub use liblumen_term::EncodingType;

use self::abi::Abi;

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

pub type TargetResult = Result<Target, String>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LinkerFlavor {
    Em,
    Gcc,
    Ld,
    Msvc,
    Lld(LldFlavor),
    PtxLinker,
}
impl fmt::Display for LinkerFlavor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Em => f.write_str("emcc"),
            Self::Gcc => f.write_str("cc"),
            Self::Ld => f.write_str("ld"),
            Self::Msvc => f.write_str("link.exe"),
            Self::Lld(_) => f.write_str("lld"),
            Self::PtxLinker => f.write_str("ptx-linker"),
        }
    }
}

pub type LinkArgs = BTreeMap<LinkerFlavor, Vec<String>>;

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
    ((LinkerFlavor::Em), "em"),
    ((LinkerFlavor::Gcc), "gcc"),
    ((LinkerFlavor::Ld), "ld"),
    ((LinkerFlavor::Msvc), "msvc"),
    ((LinkerFlavor::PtxLinker), "ptx-linker"),
    ((LinkerFlavor::Lld(LldFlavor::Wasm)), "wasm-ld"),
    ((LinkerFlavor::Lld(LldFlavor::Ld64)), "ld64.lld"),
    ((LinkerFlavor::Lld(LldFlavor::Ld)), "ld.lld"),
    ((LinkerFlavor::Lld(LldFlavor::Link)), "lld-link"),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum RelocModel {
    Default,
    Static,
    PIC,
    DynamicNoPic,
    ROPI,
    RWPI,
    #[allow(non_camel_case_types)]
    ROPI_RWPI,
}
impl FromStr for RelocModel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "default" => Ok(Self::Default),
            "static" => Ok(Self::Static),
            "pic" => Ok(Self::PIC),
            "dynamic-no-pic" => Ok(Self::DynamicNoPic),
            "ropi" => Ok(Self::ROPI),
            "rwpi" => Ok(Self::RWPI),
            "ropi-rwpi" => Ok(Self::ROPI_RWPI),
            _ => Err(()),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum CodeModel {
    #[allow(dead_code)]
    Other,
    Small,
    Kernel,
    Medium,
    Large,
    None,
}
impl FromStr for CodeModel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "small" => Ok(Self::Small),
            "kernel" => Ok(Self::Kernel),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            _ => Err(()),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Hash)]
pub enum ThreadLocalMode {
    NotThreadLocal,
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec,
}
impl FromStr for ThreadLocalMode {
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

macro_rules! supported_targets {
    ( $(($( $triple:literal, )+ $module:ident ),)+ ) => {
        $(mod $module;)+

        /// List of supported targets
        const TARGETS: &[&str] = &[$($($triple),+),+];

        fn search<S: AsRef<str>>(target: S) -> Result<Target, TargetError> {
            let target = target.as_ref();
            match target {
                $(
                    $($triple)|+ => $module::target().map_err(TargetError::Other),
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

supported_targets! {
    ("x86_64-unknown-linux-gnu", x86_64_unknown_linux_gnu),
    ("x86_64-unknown-linux-gnux32", x86_64_unknown_linux_gnux32),
    ("i686-unknown-linux-gnu", i686_unknown_linux_gnu),
    ("i586-unknown-linux-gnu", i586_unknown_linux_gnu),
    //("mips-unknown-linux-gnu", mips_unknown_linux_gnu),
    //("mips64-unknown-linux-gnuabi64", mips64_unknown_linux_gnuabi64),
    //("mips64el-unknown-linux-gnuabi64", mips64el_unknown_linux_gnuabi64),
    //("mipsisa32r6-unknown-linux-gnu", mipsisa32r6_unknown_linux_gnu),
    //("mipsisa32r6el-unknown-linux-gnu", mipsisa32r6el_unknown_linux_gnu),
    //("mipsisa64r6-unknown-linux-gnuabi64", mipsisa64r6_unknown_linux_gnuabi64),
    //("mipsisa64r6el-unknown-linux-gnuabi64", mipsisa64r6el_unknown_linux_gnuabi64),
    //("mipsel-unknown-linux-gnu", mipsel_unknown_linux_gnu),
    //("powerpc-unknown-linux-gnu", powerpc_unknown_linux_gnu),
    //("powerpc-unknown-linux-gnuspe", powerpc_unknown_linux_gnuspe),
    //("powerpc-unknown-linux-musl", powerpc_unknown_linux_musl),
    //("powerpc64-unknown-linux-gnu", powerpc64_unknown_linux_gnu),
    //("powerpc64-unknown-linux-musl", powerpc64_unknown_linux_musl),
    //("powerpc64le-unknown-linux-gnu", powerpc64le_unknown_linux_gnu),
    //("powerpc64le-unknown-linux-musl", powerpc64le_unknown_linux_musl),
    //("s390x-unknown-linux-gnu", s390x_unknown_linux_gnu),
    //("sparc-unknown-linux-gnu", sparc_unknown_linux_gnu),
    //("sparc64-unknown-linux-gnu", sparc64_unknown_linux_gnu),
    ("arm-unknown-linux-gnueabi", arm_unknown_linux_gnueabi),
    ("arm-unknown-linux-gnueabihf", arm_unknown_linux_gnueabihf),
    ("arm-unknown-linux-musleabi", arm_unknown_linux_musleabi),
    ("arm-unknown-linux-musleabihf", arm_unknown_linux_musleabihf),
    ("armv4t-unknown-linux-gnueabi", armv4t_unknown_linux_gnueabi),
    ("armv5te-unknown-linux-gnueabi", armv5te_unknown_linux_gnueabi),
    ("armv5te-unknown-linux-musleabi", armv5te_unknown_linux_musleabi),
    ("armv7-unknown-linux-gnueabi", armv7_unknown_linux_gnueabi),
    ("armv7-unknown-linux-gnueabihf", armv7_unknown_linux_gnueabihf),
    //("thumbv7neon-unknown-linux-gnueabihf", thumbv7neon_unknown_linux_gnueabihf),
    //("thumbv7neon-unknown-linux-musleabihf", thumbv7neon_unknown_linux_musleabihf),
    ("armv7-unknown-linux-musleabi", armv7_unknown_linux_musleabi),
    ("armv7-unknown-linux-musleabihf", armv7_unknown_linux_musleabihf),
    ("aarch64-unknown-linux-gnu", aarch64_unknown_linux_gnu),
    ("aarch64-unknown-linux-musl", aarch64_unknown_linux_musl),
    ("x86_64-unknown-linux-musl", x86_64_unknown_linux_musl),
    ("i686-unknown-linux-musl", i686_unknown_linux_musl),
    ("i586-unknown-linux-musl", i586_unknown_linux_musl),
    //("mips-unknown-linux-musl", mips_unknown_linux_musl),
    //("mipsel-unknown-linux-musl", mipsel_unknown_linux_musl),
    //("mips64-unknown-linux-muslabi64", mips64_unknown_linux_muslabi64),
    //("mips64el-unknown-linux-muslabi64", mips64el_unknown_linux_muslabi64),
    //("hexagon-unknown-linux-musl", hexagon_unknown_linux_musl),

    //("mips-unknown-linux-uclibc", mips_unknown_linux_uclibc),
    //("mipsel-unknown-linux-uclibc", mipsel_unknown_linux_uclibc),

    ("i686-linux-android", i686_linux_android),
    ("x86_64-linux-android", x86_64_linux_android),
    ("arm-linux-androideabi", arm_linux_androideabi),
    ("armv7-linux-androideabi", armv7_linux_androideabi),
    //("thumbv7neon-linux-androideabi", thumbv7neon_linux_androideabi),
    ("aarch64-linux-android", aarch64_linux_android),

    ("x86_64-linux-kernel", x86_64_linux_kernel),

    ("aarch64-unknown-freebsd", aarch64_unknown_freebsd),
    ("armv6-unknown-freebsd", armv6_unknown_freebsd),
    ("armv7-unknown-freebsd", armv7_unknown_freebsd),
    ("i686-unknown-freebsd", i686_unknown_freebsd),
    //("powerpc64-unknown-freebsd", powerpc64_unknown_freebsd),
    ("x86_64-unknown-freebsd", x86_64_unknown_freebsd),

    ("x86_64-unknown-dragonfly", x86_64_unknown_dragonfly),

    ("aarch64-unknown-openbsd", aarch64_unknown_openbsd),
    ("i686-unknown-openbsd", i686_unknown_openbsd),
    //("sparc64-unknown-openbsd", sparc64_unknown_openbsd),
    ("x86_64-unknown-openbsd", x86_64_unknown_openbsd),

    ("aarch64-unknown-netbsd", aarch64_unknown_netbsd),
    ("armv6-unknown-netbsd-eabihf", armv6_unknown_netbsd_eabihf),
    ("armv7-unknown-netbsd-eabihf", armv7_unknown_netbsd_eabihf),
    ("i686-unknown-netbsd", i686_unknown_netbsd),
    //("powerpc-unknown-netbsd", powerpc_unknown_netbsd),
    //("sparc64-unknown-netbsd", sparc64_unknown_netbsd),
    ("x86_64-unknown-netbsd", x86_64_unknown_netbsd),
    ("x86_64-rumprun-netbsd", x86_64_rumprun_netbsd),

    ("i686-unknown-haiku", i686_unknown_haiku),
    ("x86_64-unknown-haiku", x86_64_unknown_haiku),

    ("x86_64-apple-darwin", x86_64_apple_darwin),
    ("i686-apple-darwin", i686_apple_darwin),

    ("aarch64-fuchsia", aarch64_fuchsia),
    ("x86_64-fuchsia", x86_64_fuchsia),

    ("x86_64-unknown-l4re-uclibc", x86_64_unknown_l4re_uclibc),

    ("aarch64-unknown-redox", aarch64_unknown_redox),
    ("x86_64-unknown-redox", x86_64_unknown_redox),

    ("i386-apple-ios", i386_apple_ios),
    ("x86_64-apple-ios", x86_64_apple_ios),
    ("aarch64-apple-ios", aarch64_apple_ios),
    ("armv7-apple-ios", armv7_apple_ios),
    ("armv7s-apple-ios", armv7s_apple_ios),
    ("x86_64-apple-ios-macabi", x86_64_apple_ios_macabi),
    ("aarch64-apple-tvos", aarch64_apple_tvos),
    ("x86_64-apple-tvos", x86_64_apple_tvos),

    ("armebv7r-none-eabi", armebv7r_none_eabi),
    ("armebv7r-none-eabihf", armebv7r_none_eabihf),
    ("armv7r-none-eabi", armv7r_none_eabi),
    ("armv7r-none-eabihf", armv7r_none_eabihf),

    // `x86_64-pc-solaris` is an alias for `x86_64_sun_solaris` for backwards compatibility reasons.
    // (See <https://github.com/rust-lang/rust/issues/40531>.)
    ("x86_64-sun-solaris", "x86_64-pc-solaris", x86_64_sun_solaris),
    //("sparcv9-sun-solaris", sparcv9_sun_solaris),

    ("x86_64-pc-windows-gnu", x86_64_pc_windows_gnu),
    ("i686-pc-windows-gnu", i686_pc_windows_gnu),
    ("i686-uwp-windows-gnu", i686_uwp_windows_gnu),
    ("x86_64-uwp-windows-gnu", x86_64_uwp_windows_gnu),

    ("aarch64-pc-windows-msvc", aarch64_pc_windows_msvc),
    ("aarch64-uwp-windows-msvc", aarch64_uwp_windows_msvc),
    ("x86_64-pc-windows-msvc", x86_64_pc_windows_msvc),
    ("x86_64-uwp-windows-msvc", x86_64_uwp_windows_msvc),
    ("i686-pc-windows-msvc", i686_pc_windows_msvc),
    ("i686-uwp-windows-msvc", i686_uwp_windows_msvc),
    ("i586-pc-windows-msvc", i586_pc_windows_msvc),
    //("thumbv7a-pc-windows-msvc", thumbv7a_pc_windows_msvc),

    //("asmjs-unknown-emscripten", asmjs_unknown_emscripten),
    ("wasm32-unknown-emscripten", wasm32_unknown_emscripten),
    ("wasm32-unknown-unknown", wasm32_unknown_unknown),
    ("wasm32-wasi", wasm32_wasi),

    //("thumbv6m-none-eabi", thumbv6m_none_eabi),
    //("thumbv7m-none-eabi", thumbv7m_none_eabi),
    //("thumbv7em-none-eabi", thumbv7em_none_eabi),
    //("thumbv7em-none-eabihf", thumbv7em_none_eabihf),
    //("thumbv8m.base-none-eabi", thumbv8m_base_none_eabi),
    //("thumbv8m.main-none-eabi", thumbv8m_main_none_eabi),
    //("thumbv8m.main-none-eabihf", thumbv8m_main_none_eabihf),

    //("armv7a-none-eabi", armv7a_none_eabi),
    //("armv7a-none-eabihf", armv7a_none_eabihf),

    //("msp430-none-elf", msp430_none_elf),

    ("aarch64-unknown-cloudabi", aarch64_unknown_cloudabi),
    ("armv7-unknown-cloudabi-eabihf", armv7_unknown_cloudabi_eabihf),
    ("i686-unknown-cloudabi", i686_unknown_cloudabi),
    ("x86_64-unknown-cloudabi", x86_64_unknown_cloudabi),

    ("aarch64-unknown-hermit", aarch64_unknown_hermit),
    ("x86_64-unknown-hermit", x86_64_unknown_hermit),
    ("x86_64-unknown-hermit-kernel", x86_64_unknown_hermit_kernel),

    //("riscv32i-unknown-none-elf", riscv32i_unknown_none_elf),
    //("riscv32imc-unknown-none-elf", riscv32imc_unknown_none_elf),
    //("riscv32imac-unknown-none-elf", riscv32imac_unknown_none_elf),
    //("riscv64imac-unknown-none-elf", riscv64imac_unknown_none_elf),
    //("riscv64gc-unknown-none-elf", riscv64gc_unknown_none_elf),
    //("riscv64gc-unknown-linux-gnu", riscv64gc_unknown_linux_gnu),

    ("aarch64-unknown-none", aarch64_unknown_none),
    ("aarch64-unknown-none-softfloat", aarch64_unknown_none_softfloat),

    //("x86_64-fortanix-unknown-sgx", x86_64_fortanix_unknown_sgx),

    ("x86_64-unknown-uefi", x86_64_unknown_uefi),
    ("i686-unknown-uefi", i686_unknown_uefi),

    //("nvptx64-nvidia-cuda", nvptx64_nvidia_cuda),

    ("i686-wrs-vxworks", i686_wrs_vxworks),
    ("x86_64-wrs-vxworks", x86_64_wrs_vxworks),
    ("armv7-wrs-vxworks-eabihf", armv7_wrs_vxworks_eabihf),
    ("aarch64-wrs-vxworks", aarch64_wrs_vxworks),
    //("powerpc-wrs-vxworks", powerpc_wrs_vxworks),
    //("powerpc-wrs-vxworks-spe", powerpc_wrs_vxworks_spe),
    //("powerpc64-wrs-vxworks", powerpc64_wrs_vxworks),
}

#[repr(C)]
#[derive(PartialEq, Clone, Debug)]
pub enum Endianness {
    Native,
    Big,
    Little,
}
impl ToString for Endianness {
    fn to_string(&self) -> String {
        match *self {
            Self::Native => "native".to_string(),
            Self::Big => "big".to_string(),
            Self::Little => "little".to_string(),
        }
    }
}

/// Everything `lumen` knows about how to compile for a specific target.
///
/// Every field here must be specified, and has no default value.
#[derive(PartialEq, Clone, Debug)]
pub struct Target {
    /// Target triple to pass to LLVM.
    pub llvm_target: String,
    /// The endianness of the target
    pub target_endian: Endianness,
    /// The bit width of target pointers
    pub target_pointer_width: usize,
    /// Width of c_int type
    pub target_c_int_width: String,
    /// OS name to use for conditional compilation.
    pub target_os: String,
    /// Environment name to use for conditional compilation.
    pub target_env: String,
    /// Vendor name to use for conditional compilation.
    pub target_vendor: String,
    /// Architecture to use for ABI considerations. Valid options include: "x86",
    /// "x86_64", "arm", "aarch64", "mips", "powerpc", "powerpc64", and others.
    pub arch: String,
    /// [Data layout](http://llvm.org/docs/LangRef.html#data-layout) to pass to LLVM.
    pub data_layout: String,
    /// Linker flavor
    pub linker_flavor: LinkerFlavor,
    /// Optional settings with defaults.
    pub options: TargetOptions,
}
impl Target {
    pub fn triple(&self) -> &str {
        self.llvm_target.as_str()
    }

    pub fn search(target: &str) -> Result<Target, TargetError> {
        self::search(target)
    }

    pub fn all() -> impl Iterator<Item = String> {
        self::get_targets()
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
    pub is_builtin: bool,

    /// Term encoding
    pub encoding: EncodingType,

    /// Linker to invoke
    pub linker: Option<String>,

    /// LLD flavor
    pub lld_flavor: LldFlavor,

    /// Linker arguments that are passed *before* any user-defined libraries.
    pub pre_link_args: LinkArgs, // ... unconditionally
    pub pre_link_args_crt: LinkArgs, // ... when linking with a bundled crt
    /// Objects to link before all others, always found within the
    /// sysroot folder.
    pub pre_link_objects_exe: Vec<String>, // ... when linking an executable, unconditionally
    pub pre_link_objects_exe_crt: Vec<String>, // ... when linking an executable with a bundled crt
    pub pre_link_objects_dll: Vec<String>, // ... when linking a dylib
    /// Linker arguments that are unconditionally passed after any
    /// user-defined but before post_link_objects. Standard platform
    /// libraries that should be always be linked to, usually go here.
    pub late_link_args: LinkArgs,
    /// Linker arguments used in addition to `late_link_args` if at least one
    /// Lumen dependency is dynamically linked.
    pub late_link_args_dynamic: LinkArgs,
    /// Linker arguments used in addition to `late_link_args` if all Lumen
    /// dependencies are statically linked.
    pub late_link_args_static: LinkArgs,
    /// Objects to link after all others, always found within the
    /// sysroot folder.
    pub post_link_objects: Vec<String>, // ... unconditionally
    pub post_link_objects_crt: Vec<String>, // ... when linking with a bundled crt
    /// Linker arguments that are unconditionally passed *after* any
    /// user-defined libraries.
    pub post_link_args: LinkArgs,

    /// Environment variables to be set for the linker invocation.
    pub link_env: Vec<(String, String)>,
    /// Environment variables to be removed for the linker invocation.
    pub link_env_remove: Vec<String>,

    /// Extra arguments to pass to the external assembler (when used)
    pub asm_args: Vec<String>,

    /// Default CPU to pass to LLVM. Corresponds to `llc -mcpu=$cpu`. Defaults
    /// to "generic".
    pub cpu: String,
    /// Default target features to pass to LLVM. These features will *always* be
    /// passed, and cannot be disabled even via `-C`. Corresponds to `llc
    /// -mattr=$features`.
    pub features: String,
    /// Whether dynamic linking is available on this target. Defaults to false.
    pub dynamic_linking: bool,
    /// If dynamic linking is available, whether only cdylibs are supported.
    pub only_cdylib: bool,
    /// Whether executables are available on this target. iOS, for example, only allows static
    /// libraries. Defaults to false.
    pub executables: bool,
    /// Relocation model to use in object file. Corresponds to `llc
    /// -relocation-model=$relocation_model`. Defaults to "pic".
    pub relocation_model: String,
    /// Code model to use. Corresponds to `llc -code-model=$code_model`.
    pub code_model: Option<String>,
    /// TLS model to use. Options are "global-dynamic" (default), "local-dynamic", "initial-exec"
    /// and "local-exec". This is similar to the -ftls-model option in GCC/Clang.
    pub tls_model: String,
    /// Do not emit code that uses the "red zone", if the ABI has one. Defaults to false.
    pub disable_redzone: bool,
    /// Eliminate frame pointers from stack frames if possible. Defaults to true.
    pub eliminate_frame_pointer: bool,
    /// Emit each function in its own section. Defaults to true.
    pub function_sections: bool,
    /// String to prepend to the name of every dynamic library. Defaults to "lib".
    pub dll_prefix: String,
    /// String to append to the name of every dynamic library. Defaults to ".so".
    pub dll_suffix: String,
    /// String to append to the name of every executable.
    pub exe_suffix: String,
    /// String to prepend to the name of every static library. Defaults to "lib".
    pub staticlib_prefix: String,
    /// String to append to the name of every static library. Defaults to ".a".
    pub staticlib_suffix: String,
    /// OS family to use for conditional compilation. Valid options: "unix", "windows".
    pub target_family: Option<String>,
    /// Whether the target toolchain's ABI supports returning small structs as an integer.
    pub abi_return_struct_as_int: bool,
    /// Whether the target toolchain is like macOS's. Only useful for compiling against iOS/macOS,
    /// in particular running dsymutil and some other stuff like `-dead_strip`. Defaults to false.
    pub is_like_osx: bool,
    /// Whether the target toolchain is like Solaris's.
    /// Only useful for compiling against Illumos/Solaris,
    /// as they have a different set of linker flags. Defaults to false.
    pub is_like_solaris: bool,
    /// Whether the target toolchain is like Windows'. Only useful for compiling against Windows,
    /// only really used for figuring out how to find libraries, since Windows uses its own
    /// library naming convention. Defaults to false.
    pub is_like_windows: bool,
    pub is_like_msvc: bool,
    /// Whether the target toolchain is like Android's. Only useful for compiling against Android.
    /// Defaults to false.
    pub is_like_android: bool,
    /// Whether the target toolchain is like Emscripten's. Only useful for compiling with
    /// Emscripten toolchain.
    /// Defaults to false.
    pub is_like_emscripten: bool,
    /// Whether the target toolchain is like Fuchsia's.
    pub is_like_fuchsia: bool,
    /// Whether the linker support GNU-like arguments such as -O. Defaults to false.
    pub linker_is_gnu: bool,
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
    pub archive_format: String,
    /// Is asm!() allowed? Defaults to true.
    pub allow_asm: bool,
    /// Whether the runtime startup code requires the `main` function be passed
    /// `argc` and `argv` values.
    pub main_needs_argc_argv: bool,

    /// Flag indicating whether ELF TLS (e.g., #[thread_local]) is available for
    /// this target.
    pub has_elf_tls: bool,
    // This is mainly for easy compatibility with emscripten.
    // If we give emcc .o files that are actually .bc files it
    // will 'just work'.
    pub obj_is_bitcode: bool,

    /// Don't use this field; instead use the `.min_atomic_width()` method.
    pub min_atomic_width: Option<u64>,

    /// Don't use this field; instead use the `.max_atomic_width()` method.
    pub max_atomic_width: Option<u64>,

    /// Whether the target supports atomic CAS operations natively
    pub atomic_cas: bool,

    /// Panic strategy: "unwind" or "abort"
    pub panic_strategy: PanicStrategy,

    /// A blacklist of ABIs unsupported by the current target. Note that generic
    /// ABIs are considered to be supported on all platforms and cannot be blacklisted.
    pub abi_blacklist: Vec<Abi>,

    /// Whether or not linking dylibs to a static CRT is allowed.
    pub crt_static_allows_dylibs: bool,
    /// Whether or not the CRT is statically linked by default.
    pub crt_static_default: bool,
    /// Whether or not crt-static is respected by the compiler (or is a no-op).
    pub crt_static_respected: bool,

    /// Whether or not stack probes (__rust_probestack) are enabled
    pub stack_probes: bool,

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

    /// The codegen backend to use for this target, typically "llvm"
    pub codegen_backend: String,

    /// The default visibility for symbols in this target should be "hidden"
    /// rather than "default"
    pub default_hidden_visibility: bool,

    /// Whether a .debug_gdb_scripts section will be added to the output object file
    pub emit_debug_gdb_scripts: bool,

    /// Whether or not to unconditionally `uwtable` attributes on functions,
    /// typically because the platform needs to unwind for things like stack
    /// unwinders.
    pub requires_uwtable: bool,

    /// Whether or not SIMD types are passed by reference in the Rust ABI,
    /// typically required if a target can be compiled with a mixed set of
    /// target features. This is `true` by default, and `false` for targets like
    /// wasm32 where the whole program either has simd or not.
    pub simd_types_indirect: bool,

    /// Pass a list of symbol which should be exported in the dylib to the linker.
    pub limit_rdylib_exports: bool,

    /// If set, have the linker export exactly these symbols, instead of using
    /// the usual logic to figure this out from the crate itself.
    pub override_export_symbols: Option<Vec<String>>,

    /// Determines how or whether the MergeFunctions LLVM pass should run for
    /// this target. Either "disabled", "trampolines", or "aliases".
    /// The MergeFunctions pass is generally useful, but some targets may need
    /// to opt out. The default is "aliases".
    ///
    /// Workaround for: https://github.com/rust-lang/rust/issues/57356
    pub merge_functions: MergeFunctions,

    /// Use platform dependent mcount function
    pub target_mcount: String,

    /// LLVM ABI name, corresponds to the '-mabi' parameter available in multilib C compilers
    pub llvm_abiname: String,

    /// Whether or not RelaxElfRelocation flag will be passed to the linker
    pub relax_elf_relocations: bool,

    /// Additional arguments to pass to LLVM, similar to the `-C llvm-args` codegen option.
    pub llvm_args: Vec<String>,
}

impl Default for TargetOptions {
    /// Creates a set of "sane defaults" for any target. This is still
    /// incomplete, and if used for compilation, will certainly not work.
    fn default() -> TargetOptions {
        TargetOptions {
            is_builtin: false,
            encoding: EncodingType::Default,
            linker: option_env!("CFG_DEFAULT_LINKER").map(|s| s.to_string()),
            lld_flavor: LldFlavor::Ld,
            pre_link_args: LinkArgs::new(),
            pre_link_args_crt: LinkArgs::new(),
            post_link_args: LinkArgs::new(),
            asm_args: Vec::new(),
            cpu: "generic".to_string(),
            features: String::new(),
            dynamic_linking: false,
            only_cdylib: false,
            executables: false,
            relocation_model: "pic".to_string(),
            code_model: None,
            tls_model: "global-dynamic".to_string(),
            disable_redzone: false,
            eliminate_frame_pointer: true,
            function_sections: true,
            dll_prefix: "lib".to_string(),
            dll_suffix: ".so".to_string(),
            exe_suffix: String::new(),
            staticlib_prefix: "lib".to_string(),
            staticlib_suffix: ".a".to_string(),
            target_family: None,
            abi_return_struct_as_int: false,
            is_like_osx: false,
            is_like_solaris: false,
            is_like_windows: false,
            is_like_android: false,
            is_like_emscripten: false,
            is_like_msvc: false,
            is_like_fuchsia: false,
            linker_is_gnu: false,
            allows_weak_linkage: true,
            has_rpath: false,
            no_default_libraries: true,
            position_independent_executables: false,
            needs_plt: false,
            relro_level: RelroLevel::None,
            pre_link_objects_exe: Vec::new(),
            pre_link_objects_exe_crt: Vec::new(),
            pre_link_objects_dll: Vec::new(),
            post_link_objects: Vec::new(),
            post_link_objects_crt: Vec::new(),
            late_link_args: LinkArgs::new(),
            late_link_args_dynamic: LinkArgs::new(),
            late_link_args_static: LinkArgs::new(),
            link_env: Vec::new(),
            link_env_remove: Vec::new(),
            archive_format: "gnu".to_string(),
            main_needs_argc_argv: true,
            allow_asm: true,
            has_elf_tls: false,
            obj_is_bitcode: false,
            min_atomic_width: None,
            max_atomic_width: None,
            atomic_cas: true,
            panic_strategy: PanicStrategy::Unwind,
            abi_blacklist: vec![],
            crt_static_allows_dylibs: false,
            crt_static_default: false,
            crt_static_respected: false,
            stack_probes: false,
            min_global_align: None,
            default_codegen_units: None,
            trap_unreachable: true,
            requires_lto: false,
            singlethread: false,
            no_builtins: false,
            codegen_backend: "llvm".to_string(),
            default_hidden_visibility: false,
            emit_debug_gdb_scripts: true,
            requires_uwtable: false,
            simd_types_indirect: true,
            limit_rdylib_exports: true,
            override_export_symbols: None,
            merge_functions: MergeFunctions::Aliases,
            target_mcount: "mcount".to_string(),
            llvm_abiname: "".to_string(),
            relax_elf_relocations: false,
            llvm_args: vec![],
        }
    }
}

impl Target {
    /// Given a function ABI, turn it into the correct ABI for this target.
    pub fn adjust_abi(&self, abi: Abi) -> Abi {
        match abi {
            Abi::System => {
                if self.options.is_like_windows && self.arch == "x86" {
                    Abi::Stdcall
                } else {
                    Abi::C
                }
            }
            // These ABI kinds are ignored on non-x86 Windows targets.
            // See https://docs.microsoft.com/en-us/cpp/cpp/argument-passing-and-naming-conventions
            // and the individual pages for __stdcall et al.
            Abi::Stdcall | Abi::Fastcall | Abi::Vectorcall | Abi::Thiscall => {
                if self.options.is_like_windows && self.arch != "x86" {
                    Abi::C
                } else {
                    abi
                }
            }
            Abi::EfiApi => {
                if self.arch == "x86_64" {
                    Abi::Win64
                } else {
                    Abi::C
                }
            }
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
            .unwrap_or_else(|| self.target_pointer_width as u64)
    }

    pub fn is_abi_supported(&self, abi: Abi) -> bool {
        abi.generic() || !self.options.abi_blacklist.contains(&abi)
    }
}

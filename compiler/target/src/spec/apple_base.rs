use std::borrow::Cow;
use std::env;

use crate::spec::{
    DebuginfoKind, FramePointer, LinkArgs, LinkerFlavor, LldFlavor, SplitDebugInfo, TargetOptions,
};

fn pre_link_args(os: &'static str, arch: &'static str, abi: &'static str) -> LinkArgs {
    let platform_name: Cow<'static, str> = match abi {
        "sim" => format!("{}-simulator", os).into(),
        "macabi" => "mac-catalyst".into(),
        _ => os.into(),
    };

    let platform_version: Cow<'static, str> = match os.as_ref() {
        "ios" => ios_lld_platform_version(),
        "tvos" => tvos_lld_platform_version(),
        "watchos" => watchos_lld_platform_version(),
        "macos" => macos_lld_platform_version(arch),
        _ => unreachable!(),
    }
    .into();

    let mut args = TargetOptions::link_args(
        LinkerFlavor::Lld(LldFlavor::Ld64),
        &["-arch", arch, "-platform_version"],
    );
    // Manually add owned args unsupported by link arg building helpers.
    args.entry(LinkerFlavor::Lld(LldFlavor::Ld64))
        .or_default()
        .extend([platform_name, platform_version.clone(), platform_version]);
    if abi != "macabi" {
        super::add_link_args(&mut args, LinkerFlavor::Gcc, &["-arch", arch]);
    }

    args
}

pub fn opts(os: &'static str, arch: &'static str, abi: &'static str) -> TargetOptions {
    // ELF TLS is only available in macOS 10.7+. If you try to compile for 10.6
    // either the linker will complain if it is used or the binary will end up
    // segfaulting at runtime when run on 10.6. Rust by default supports macOS
    // 10.7+, but there is a standard environment variable,
    // MACOSX_DEPLOYMENT_TARGET, which is used to signal targeting older
    // versions of macOS. For example compiling on 10.10 with
    // MACOSX_DEPLOYMENT_TARGET set to 10.6 will cause the linker to generate
    // warnings about the usage of ELF TLS.
    //
    // Here we detect what version is being requested, defaulting to 10.7. ELF
    // TLS is flagged as enabled if it looks to be supported. The architecture
    // only matters for default deployment target which is 11.0 for ARM64 and
    // 10.7 for everything else.
    let has_thread_local = macos_deployment_target("x86_64") >= (10, 7);

    TargetOptions {
        os: os.into(),
        vendor: "apple".into(),
        // macOS has -dead_strip, which doesn't rely on function_sections
        function_sections: false,
        dynamic_linking: true,
        pre_link_args: pre_link_args(os, arch, abi),
        linker_is_gnu: false,
        families: vec!["unix".into()],
        is_like_osx: true,
        default_dwarf_version: 2,
        frame_pointer: FramePointer::Always,
        has_rpath: true,
        dll_suffix: ".dylib".into(),
        archive_format: "darwin".into(),
        has_thread_local,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        eh_frame_header: false,
        lld_flavor: LldFlavor::Ld64,

        debuginfo_kind: DebuginfoKind::DwarfDsym,
        // The historical default for macOS targets is to run `dsymutil` which
        // generates a packed version of debuginfo split from the main file.
        split_debuginfo: SplitDebugInfo::Packed,
        supported_split_debuginfo: Cow::Borrowed(&[
            SplitDebugInfo::Packed,
            SplitDebugInfo::Unpacked,
            SplitDebugInfo::Off,
        ]),

        // This environment variable is pretty magical but is intended for
        // producing deterministic builds. This was first discovered to be used
        // by the `ar` tool as a way to control whether or not mtime entries in
        // the archive headers were set to zero or not. It appears that
        // eventually the linker got updated to do the same thing and now reads
        // this environment variable too in recent versions.
        //
        // For some more info see the commentary on #47086
        link_env: vec![("ZERO_AR_DATE".into(), "1".into())],

        ..Default::default()
    }
}

fn deployment_target(var_name: &str) -> Option<(u32, u32)> {
    let deployment_target = env::var(var_name).ok();
    deployment_target
        .as_ref()
        .and_then(|s| s.split_once('.'))
        .and_then(|(a, b)| {
            a.parse::<u32>()
                .and_then(|a| b.parse::<u32>().map(|b| (a, b)))
                .ok()
        })
}

fn macos_default_deployment_target(arch: &str) -> (u32, u32) {
    if arch == "arm64" {
        (11, 0)
    } else {
        (10, 7)
    }
}

fn macos_deployment_target(arch: &str) -> (u32, u32) {
    deployment_target("MACOSX_DEPLOYMENT_TARGET")
        .unwrap_or_else(|| macos_default_deployment_target(arch))
}

fn macos_lld_platform_version(arch: &str) -> String {
    let (major, minor) = macos_deployment_target(arch);
    format!("{}.{}", major, minor)
}

pub fn macos_llvm_target(arch: &str) -> Cow<'static, str> {
    let (major, minor) = macos_deployment_target(arch);
    format!("{}-apple-macosx{}.{}.0", arch, major, minor).into()
}

pub fn macos_link_env_remove() -> Vec<Cow<'static, str>> {
    let mut env_remove = Vec::with_capacity(2);
    // Remove the `SDKROOT` environment variable if it's clearly set for the wrong platform, which
    // may occur when we're linking a custom build script while targeting iOS for example.
    if let Ok(sdkroot) = env::var("SDKROOT") {
        if sdkroot.contains("iPhoneOS.platform") || sdkroot.contains("iPhoneSimulator.platform") {
            env_remove.push("SDKROOT".into())
        }
    }
    // Additionally, `IPHONEOS_DEPLOYMENT_TARGET` must not be set when using the Xcode linker at
    // "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ld",
    // although this is apparently ignored when using the linker at "/usr/bin/ld".
    env_remove.push("IPHONEOS_DEPLOYMENT_TARGET".into());
    env_remove
}

fn ios_deployment_target() -> (u32, u32) {
    deployment_target("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or((7, 0))
}

pub fn ios_llvm_target(arch: &str) -> String {
    // Modern iOS tooling extracts information about deployment target
    // from LC_BUILD_VERSION. This load command will only be emitted when
    // we build with a version specific `llvm_target`, with the version
    // set high enough. Luckily one LC_BUILD_VERSION is enough, for Xcode
    // to pick it up (since std and core are still built with the fallback
    // of version 7.0 and hence emit the old LC_IPHONE_MIN_VERSION).
    let (major, minor) = ios_deployment_target();
    format!("{}-apple-ios{}.{}.0", arch, major, minor)
}

fn ios_lld_platform_version() -> String {
    let (major, minor) = ios_deployment_target();
    format!("{}.{}", major, minor)
}

pub fn ios_sim_llvm_target(arch: &str) -> String {
    let (major, minor) = ios_deployment_target();
    format!("{}-apple-ios{}.{}.0-simulator", arch, major, minor)
}

fn tvos_deployment_target() -> (u32, u32) {
    deployment_target("TVOS_DEPLOYMENT_TARGET").unwrap_or((7, 0))
}

fn tvos_lld_platform_version() -> String {
    let (major, minor) = tvos_deployment_target();
    format!("{}.{}", major, minor)
}

fn watchos_deployment_target() -> (u32, u32) {
    deployment_target("WATCHOS_DEPLOYMENT_TARGET").unwrap_or((5, 0))
}

fn watchos_lld_platform_version() -> String {
    let (major, minor) = watchos_deployment_target();
    format!("{}.{}", major, minor)
}

pub fn watchos_sim_llvm_target(arch: &str) -> String {
    let (major, minor) = watchos_deployment_target();
    format!("{}-apple-watchos{}.{}.0-simulator", arch, major, minor)
}

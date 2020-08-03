use super::apple_sdk_base::{opts, AppleOS, Arch};
use crate::spec::{EncodingType, Endianness, LinkerFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    let base = opts(Arch::X86_64, AppleOS::iOS)?;
    Ok(Target {
        llvm_target: "x86_64-apple-ios".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),
        target_os: "ios".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            encoding: EncodingType::Encoding64Nanboxed,
            max_atomic_width: Some(64),
            stack_probes: true,
            ..base
        },
    })
}

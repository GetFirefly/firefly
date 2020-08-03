use super::apple_sdk_base::{opts, AppleOS, Arch};
use crate::spec::{Endianness, LinkerFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    let base = opts(Arch::Arm64, AppleOS::tvOS)?;
    Ok(Target {
        llvm_target: "arm64-apple-tvos".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        target_os: "tvos".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a7".to_string(),
            eliminate_frame_pointer: false,
            max_atomic_width: Some(128),
            unsupported_abis: super::arm_base::unsupported_abis(),
            forces_embed_bitcode: true,
            ..base
        },
    })
}

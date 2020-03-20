use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult, Endianness, EncodingType};
use super::apple_ios_base::{opts, Arch};

pub fn target() -> TargetResult {
    let base = opts(Arch::X86_64_macabi)?;
    Ok(Target {
        llvm_target: "x86_64-apple-ios13.0-macabi".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        arch: "x86_64".to_string(),
        target_os: "ios".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            encoding: EncodingType::Encoding64Nanboxed,
            max_atomic_width: Some(64),
            stack_probes: true,
            .. base
        }
    })
}

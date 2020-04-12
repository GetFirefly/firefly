use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult, Endianness, EncodingType};

pub fn target() -> TargetResult {
    let mut base = super::l4re_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);

    Ok(Target {
        llvm_target: "x86_64-unknown-l4re-uclibc".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),
        target_os: "l4re".to_string(),
        target_env: "uclibc".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Ld,
        options: TargetOptions {
            encoding: EncodingType::Encoding64Nanboxed,
            ..base
        },
    })
}

use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions, TargetResult, Endianness, EncodingType};

pub fn target() -> TargetResult {
    let mut base = super::hermit_kernel_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);
    base.features =
        "-mmx,-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-3dnow,-3dnowa,-avx,-avx2,+soft-float"
            .to_string();
    base.stack_probes = true;

    Ok(Target {
        llvm_target: "x86_64-unknown-hermit".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),
        target_os: "hermit".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        options: TargetOptions {
            encoding: EncodingType::Encoding64Nanboxed,
            ..base
        },
    })
}

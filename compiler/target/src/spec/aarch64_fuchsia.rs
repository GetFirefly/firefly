use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions, TargetResult, Endianness};

pub fn target() -> TargetResult {
    let mut base = super::fuchsia_base::opts();
    base.max_atomic_width = Some(128);

    Ok(Target {
        llvm_target: "aarch64-fuchsia".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        target_os: "fuchsia".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        options: TargetOptions { abi_blacklist: super::arm_base::abi_blacklist(), ..base },
    })
}

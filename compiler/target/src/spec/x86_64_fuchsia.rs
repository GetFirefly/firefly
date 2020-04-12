use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetResult, Endianness};

pub fn target() -> TargetResult {
    let mut base = super::fuchsia_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);
    base.stack_probes = true;

    Ok(Target {
        llvm_target: "x86_64-fuchsia".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),
        target_os: "fuchsia".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        options: base,
    })
}

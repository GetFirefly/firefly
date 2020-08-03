// Targets the Big endian Cortex-R4/R5 processor (ARMv7-R)

use crate::spec::{
    Endianness, LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, Target, TargetOptions,
    TargetResult,
};
use std::default::Default;

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "armebv7r-unknown-none-eabi".to_string(),
        target_endian: Endianness::Big,
        target_pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: "".to_string(),
        target_vendor: "".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),

        options: TargetOptions {
            executables: true,
            linker: Some("lumen-lld".to_owned()),
            relocation_model: RelocModel::Static,
            panic_strategy: PanicStrategy::Abort,
            max_atomic_width: Some(32),
            unsupported_abis: super::arm_base::unsupported_abis(),
            emit_debug_gdb_scripts: false,
            ..Default::default()
        },
    })
}

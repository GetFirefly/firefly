// Targets the Cortex-R4F/R5F processor (ARMv7-R)

use std::default::Default;
use crate::spec::{LinkerFlavor, LldFlavor, PanicStrategy, Target, TargetOptions, TargetResult, Endianness};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "armebv7r-unknown-none-eabihf".to_string(),
        target_endian: Endianness::Big,
        target_pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),

        options: TargetOptions {
            executables: true,
            linker: Some("rust-lld".to_owned()),
            relocation_model: "static".to_string(),
            panic_strategy: PanicStrategy::Abort,
            features: "+vfp3,-d32,-fp16".to_string(),
            max_atomic_width: Some(32),
            abi_blacklist: super::arm_base::abi_blacklist(),
            emit_debug_gdb_scripts: false,
            .. Default::default()
        },
    })
}

use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult, Endianness};

pub fn target() -> TargetResult {
    let mut base = super::android_base::opts();
    // https://developer.android.com/ndk/guides/abis.html#armeabi
    base.features = "+strict-align,+v5te".to_string();
    base.max_atomic_width = Some(32);

    Ok(Target {
        llvm_target: "arm-linux-androideabi".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "android".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions { abi_blacklist: super::arm_base::abi_blacklist(), ..base },
    })
}

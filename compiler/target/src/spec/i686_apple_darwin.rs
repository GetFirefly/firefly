use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult, Endianness};

pub fn target() -> TargetResult {
    let mut base = super::apple_base::opts();
    base.cpu = "yonah".to_string();
    base.max_atomic_width = Some(64);
    base.pre_link_args.insert(LinkerFlavor::Gcc, vec!["-m32".to_string()]);
    base.link_env_remove.extend(super::apple_base::macos_link_env_remove());
    base.stack_probes = true;
    base.eliminate_frame_pointer = false;

    // Clang automatically chooses a more specific target based on
    // MACOSX_DEPLOYMENT_TARGET.  To enable cross-language LTO to work
    // correctly, we do too.
    let arch = "i686";
    let llvm_target = super::apple_base::macos_llvm_target(&arch);

    Ok(Target {
        llvm_target: llvm_target,
        target_endian: Endianness::Little,
        target_pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128".to_string(),
        arch: "x86".to_string(),
        target_os: "macos".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            target_mcount: "\u{1}mcount".to_string(),
            .. base
        },
    })
}

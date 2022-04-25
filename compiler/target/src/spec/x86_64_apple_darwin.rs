use crate::spec::{EncodingType, FramePointer, LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::apple_base::opts("macos");
    base.cpu = "core2".into();
    base.max_atomic_width = Some(128); // core2 support cmpxchg16b
    base.frame_pointer = FramePointer::Always;
    base.pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec!["-m64".into(), "-arch".into(), "x86_64".into()],
    );
    let mut link_env_remove = super::apple_base::macos_link_env_remove();
    base.link_env_remove.append(&mut link_env_remove);
    // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
    base.stack_probes = false;

    // Clang automatically chooses a more specific target based on
    // MACOSX_DEPLOYMENT_TARGET.  To enable cross-language LTO to work
    // correctly, we do too.
    let arch = "x86_64";
    let llvm_target = super::apple_base::macos_llvm_target(&arch);

    Target {
        llvm_target: llvm_target.into(),
        pointer_width: 64,
        data_layout: "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .into(),
        arch: arch.into(),
        options: TargetOptions {
            encoding: EncodingType::Encoding64Nanboxed,
            mcount: "\u{1}mcount".into(),
            ..base
        },
    }
}

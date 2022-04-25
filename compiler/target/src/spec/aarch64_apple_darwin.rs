use crate::spec::{FramePointer, LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::apple_base::opts("macos");
    base.cpu = "apple-a14".into();
    base.max_atomic_width = Some(128);

    base.pre_link_args
        .insert(LinkerFlavor::Gcc, vec!["-arch".into(), "arm64".into()]);

    base.link_env_remove
        .extend(super::apple_base::macos_link_env_remove());

    // Clang automatically chooses a more specific target based on
    // MACOSX_DEPLOYMENT_TARGET.  To enable cross-language LTO to work
    // correctly, we do too.
    let llvm_target = super::apple_base::macos_llvm_target("arm64");

    Target {
        llvm_target: llvm_target.into(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            mcount: "\u{1}mcount".into(),
            frame_pointer: FramePointer::NonLeaf,
            ..base
        },
    }
}

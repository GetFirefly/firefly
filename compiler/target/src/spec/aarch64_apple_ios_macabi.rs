use super::apple_sdk_base::{opts, Arch};
use crate::spec::{FramePointer, LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let llvm_target = "arm64-apple-ios14.0-macabi";

    let mut base = opts("ios", Arch::Arm64_macabi);
    base.add_pre_link_args(LinkerFlavor::Gcc, &["-target", llvm_target]);

    Target {
        llvm_target: llvm_target.into(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+apple-a12".into(),
            max_atomic_width: Some(128),
            forces_embed_bitcode: true,
            frame_pointer: FramePointer::NonLeaf,
            // Taken from a clang build on Xcode 11.4.1.
            // These arguments are not actually invoked - they just have
            // to look right to pass App Store validation.
            bitcode_llvm_cmdline: "-triple\0\
                arm64-apple-ios14.0-macabi\0\
                -emit-obj\0\
                -disable-llvm-passes\0\
                -Os\0"
                .into(),
            ..base
        },
    }
}

use super::apple_sdk_base::{opts, Arch};
use crate::spec::{StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let base = opts("tvos", Arch::X86_64);
    Target {
        llvm_target: "x86_64-apple-tvos".into(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: TargetOptions {
            max_atomic_width: Some(64),
            // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
            stack_probes: StackProbeType::Call,
            ..base
        },
    }
}

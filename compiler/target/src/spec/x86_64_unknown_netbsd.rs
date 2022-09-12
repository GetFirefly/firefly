use crate::spec::{LinkerFlavor, StackProbeType, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::netbsd_base::opts();
    base.cpu = "x86-64".into();
    base.max_atomic_width = Some(64);
    base.pre_link_args
        .entry(LinkerFlavor::Gcc)
        .or_default()
        .push("-m64".into());
    // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
    base.stack_probes = StackProbeType::Call;

    Target {
        llvm_target: "x86_64-unknown-netbsd".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .into(),
        arch: "x86_64".into(),
        options: TargetOptions {
            mcount: "__mcount".into(),
            ..base
        },
    }
}

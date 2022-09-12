use crate::spec::{LinkerFlavor, StackProbeType, Target};

pub fn target() -> Target {
    let mut base = super::linux_gnu_base::opts();
    base.cpu = "x86-64".into();
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Gcc, &["-m64"]);
    // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
    base.stack_probes = StackProbeType::Call;
    base.static_position_independent_executables = true;

    Target {
        llvm_target: "x86_64-unknown-linux-gnu".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .into(),
        arch: "x86_64".into(),
        options: base,
    }
}

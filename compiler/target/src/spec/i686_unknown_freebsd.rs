use crate::spec::{LinkerFlavor, StackProbeType, Target};

pub fn target() -> Target {
    let mut base = super::freebsd_base::opts();
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    let pre_link_args = base.pre_link_args.entry(LinkerFlavor::Gcc).or_default();
    pre_link_args.push("-m32".into());
    pre_link_args.push("-Wl,-znotext".into());
    // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
    base.stack_probes = StackProbeType::Call;

    Target {
        llvm_target: "i686-unknown-freebsd".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}

use crate::spec::{RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "dragonfly".into(),
        dynamic_linking: true,
        families: vec!["unix".into()],
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        default_dwarf_version: 2,
        ..Default::default()
    }
}
